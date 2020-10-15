#/////////////////////////#
#   import dependencies   #
#/////////////////////////#


import numpy as np
import pandas as pd
from datetime import datetime
import pandas_datareader as dr 
import pandas_datareader.data as web
from matplotlib import pyplot as plt
import seaborn as sns


import plotly.graph_objects as go
import plotly.express as px

import tensorflow as tf

import IPython
import IPython.display


#/////////////////////////#
#        Formatting       #
#/////////////////////////#


sns.set(style='darkgrid', context='paper', palette='Dark2')
sns.despine()


#/////////////////////////#
#        Timestamps       #
#/////////////////////////#



start_date = datetime(2017,10,8)
end_date = datetime(2020,10,8)

months = ["August","September","October"]
months_timestamp = [datetime(2020,i,1)for i in range(8,11)]

#/////////////////////////#
#    select tickers       #
#/////////////////////////#

# Market symbols for our stocks
tickers = ["MSFT","INTC","AMZN","AAPL","GOOG","TSLA","NFLX","PBW","ZM","LVGO","PLUG","NVDA"]

num_symbols = len(tickers)



#/////////////////////////#
#        load data        #
#/////////////////////////#


# GET data using pandas datareader via yahoo finance API
stock_data = web.DataReader([symbol for symbol in tickers], 'yahoo', start_date, end_date)

# stock_data.to_csv('stock_dat.csv')


# stock_data = pd.read_csv('stock_dat.csv')


#//////////////////////////#
#        preprocess        #
#//////////////////////////#

# Drop extra index
stock_data = stock_data.droplevel(level = 'Symbols',axis = 1)


# Drop Adjusted Close and Dates(keep the dates in a seperate Series)
# reindex
stock_data.reset_index(inplace = True)
# pop dates
dates = stock_data.pop('Date')
# drop Adj Close
stock_data.drop(columns=['Adj Close','Volume'],inplace = True)
plot_col = 'Open:MSFT'
metrics = ['Close','High','Low','Open']

cols = []

for metric in metrics:
    for ticker in tickers:
        cols.append("{}:{}".format(metric,ticker))

stock_data.columns = cols
print(stock_data.head())



# get stats
print(stock_data.describe())

corr = stock_data.corr()




#///////////////////////////////#
#        train test split       #
#///////////////////////////////#

column_indices = {name: i for i, name in enumerate(stock_data.columns)}

n = len(stock_data)
train_df = stock_data[0:int(n*0.7)]
val_df = stock_data[int(n*0.7):int(n*0.9)]
test_df= stock_data[int(n*0.9):]

num_features = stock_data.shape[1]
print(num_features)

# Normalise data

train_mean = train_df.mean()
train_std = train_df.std()

train_df = (train_df - train_mean) / train_std
val_df = (val_df - train_mean) / train_std
test_df = (test_df - train_mean) / train_std

df_std = (stock_data - train_mean) / train_std
df_std = df_std.melt(var_name='Column', value_name='Normalized')
plt.figure(figsize=(12, 6))
ax = sns.violinplot(x='Column', y='Normalized', data=df_std)
_ = ax.set_xticklabels(stock_data.keys(), rotation=90)

plt.show()
print(df_std)


#///////////////////////////////#
#      Data Windowing Class     #
#///////////////////////////////#

class WindowGenerator():
    def __init__(self, input_width, label_width, shift,
               train_df=train_df, val_df=val_df, test_df=test_df,
               label_columns=None):
    # Store the raw data.
        self.train_df = train_df
        self.val_df = val_df
        self.test_df = test_df

        # Work out the label column indices.
        self.label_columns = label_columns
        if label_columns is not None:
            self.label_columns_indices = {name: i for i, name in
                                        enumerate(label_columns)}
        self.column_indices = {name: i for i, name in
                            enumerate(train_df.columns)}

        # Work out the window parameters.
        self.input_width = input_width
        self.label_width = label_width
        self.shift = shift

        self.total_window_size = input_width + shift

        self.input_slice = slice(0, input_width)
        self.input_indices = np.arange(self.total_window_size)[self.input_slice]

        self.label_start = self.total_window_size - self.label_width
        self.labels_slice = slice(self.label_start, None)
        self.label_indices = np.arange(self.total_window_size)[self.labels_slice]

    def __repr__(self):
        return '\n'.join([
            f'Total window size: {self.total_window_size}',
            f'Input indices: {self.input_indices}',
            f'Label indices: {self.label_indices}',
            f'Label column name(s): {self.label_columns}'])

    #//////////////////#
    #      METHODS     #
    #//////////////////#

    # ___________________________________________________________
    # Getter Methods
    # -----------------------------------------------------------
    @property
    def train(self):
        return self.make_dataset(self.train_df)

    @property
    def val(self):
        return self.make_dataset(self.val_df)

    @property
    def test(self):
        return self.make_dataset(self.test_df)

    @property
    def example(self):
        """Get and cache an example batch of `inputs, labels` for plotting."""
        result = getattr(self, '_example', None)
        if result is None:
            # No example batch was found, so get one from the `.train` dataset
            result = next(iter(self.train))
            # And cache it for next time
            self._example = result
        return result


    #////////////////////////#
    #      CLASS METHODS     #
    #////////////////////////#
    
    # ___________________________________________________________
    # Split window into inputs and labels
    # -----------------------------------------------------------

    def split_window(self, features):
        inputs = features[:, self.input_slice, :]
        labels = features[:, self.labels_slice, :]
        if self.label_columns is not None:
            labels = tf.stack(
                [labels[:, :, self.column_indices[name]] for name in self.label_columns],
                axis=-1)

        # Slicing doesn't preserve static shape information, so set the shapes
        # manually. This way the `tf.data.Datasets` are easier to inspect.
        inputs.set_shape([None, self.input_width, None])
        labels.set_shape([None, self.label_width, None])

        return inputs, labels

    # ___________________________________________________________
    # Make Datasets
    # -----------------------------------------------------------
    def make_dataset(self, data):
        data = np.array(data, dtype=np.float32)
        ds = tf.keras.preprocessing.timeseries_dataset_from_array(
            data=data,
            targets=None,
            sequence_length=self.total_window_size,
            sequence_stride=1,
            shuffle=True,
            batch_size=32,)

        ds = ds.map(self.split_window)

        return ds

    # ___________________________________________________________
    # Plot
    # -----------------------------------------------------------

    def plot(self, model=None, plot_col=plot_col, max_subplots=3):
        inputs, labels = self.example
        plt.figure(figsize=(12, 8))
        plot_col_index = self.column_indices[plot_col]
        max_n = min(max_subplots, len(inputs))
        for n in range(max_n):
            plt.subplot(3, 1, n+1)
            plt.ylabel(f'{plot_col} [normed]')
            plt.plot(self.input_indices, inputs[n, :, plot_col_index],
                    label='Inputs', marker='.', zorder=-10)

            if self.label_columns:
                label_col_index = self.label_columns_indices.get(plot_col, None)
            else:
                label_col_index = plot_col_index

            if label_col_index is None:
                continue

            plt.scatter(self.label_indices, labels[n, :, label_col_index],
                        edgecolors='k', label='Labels', c='#2ca02c', s=64)
            if model is not None:
                predictions = model(inputs)
                plt.scatter(self.label_indices, predictions[n, :, label_col_index],
                        marker='X', edgecolors='k', label='Predictions',
                        c='#ff7f0e', s=64)

            if n == 0:
                plt.legend()
            if model:
                plt.title(model.id)
            plt.xlabel('Time [h]')
        
        plt.title(plot_col)
        plt.show()



# ___________________________________________________________
# ___________________________________________________________
# WINDOWS
# -----------------------------------------------------------
# -----------------------------------------------------------




# single
single_step_window = WindowGenerator(
    input_width=1, label_width=1, shift=1,
    label_columns=[plot_col])
print(single_step_window)


# wide (1 month)
wide_window = WindowGenerator(
    input_width=21, label_width=21, shift=1,
    label_columns=[plot_col])

print(wide_window)


# ___________________________________________________________
# ___________________________________________________________
# MODELS
# -----------------------------------------------------------
# -----------------------------------------------------------

#////////////////////////#
#         Baseline       #
#////////////////////////#

# Baseline Class Predicts no change
class Baseline(tf.keras.Model):
  def __init__(self, label_index=None,ID = 'Model'):
    super().__init__()
    self.label_index = label_index
    self.id = ID

  def call(self, inputs):
    if self.label_index is None:
      return inputs
    result = inputs[:, :, self.label_index]
    return result[:, :, tf.newaxis]

#////////////////////////#
#         Sequential        #
#////////////////////////#


# sequential custom class
class Sequential(tf.keras.Sequential):

    def __init__(self, layers=None, name=None,ID = 'Model'):
        super().__init__(layers=layers, name=name)
        self.id = ID

    def compile_and_fit(self, window, patience=4,MAX_EPOCHS = 200,baseline = None):
        early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                            patience=patience,
                                                            mode='min',
                                                            restore_best_weights = True,
                                                            baseline = baseline)

        self.compile(loss=tf.losses.MeanSquaredError(),
                        optimizer=tf.optimizers.Adam(),
                        metrics=[tf.metrics.MeanAbsoluteError()])

        history = self.fit(window.train, epochs=MAX_EPOCHS,
                            validation_data=window.val,
                            callbacks=[early_stopping])
        return history

def compile_and_fit(model,window, patience=4,MAX_EPOCHS = 200,baseline = None):
        early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                            patience=patience,
                                                            mode='min',
                                                            restore_best_weights = True,
                                                            baseline = baseline)

        model.compile(loss=tf.losses.MeanSquaredError(),
                        optimizer=tf.optimizers.Adam(),
                        metrics=[tf.metrics.MeanAbsoluteError()])

        history = model.fit(window.train, epochs=MAX_EPOCHS,
                            validation_data=window.val,
                            callbacks=[early_stopping])
        return history


# ___________________________________________________________
# ___________________________________________________________
# EXECTUTION (SINGLE STEP)
# -----------------------------------------------------------
# -----------------------------------------------------------


#////////////////////////#
#         Baseline       #
#////////////////////////#


baseline = Baseline(label_index=column_indices[plot_col])

baseline.compile(loss=tf.losses.MeanSquaredError(),
                 metrics=[tf.metrics.MeanAbsoluteError()])

val_performance = {}
performance = {}
val_performance['Baseline'] = baseline.evaluate(single_step_window.val)
performance['Baseline'] = baseline.evaluate(single_step_window.test, verbose=0)

wide_window.plot(baseline)


#////////////////////////#
#         linear         #
#////////////////////////#

linear = Sequential([
    tf.keras.layers.Dense(units=1)
])


history = linear.compile_and_fit( single_step_window)

val_performance['Linear'] = linear.evaluate(single_step_window.val)
performance['Linear'] = linear.evaluate(single_step_window.test, verbose=0)


wide_window.plot(linear)
plt.show()


plt.bar(x = range(len(train_df.columns)),
        height=linear.layers[0].kernel[:,0].numpy())
axis = plt.gca()
axis.set_xticks(range(len(train_df.columns)))
_ = axis.set_xticklabels(train_df.columns, rotation=90)
plt.show()



#////////////////////////#
#         multidense     #
#////////////////////////#

dense = Sequential([
    tf.keras.layers.Dense(units=64, activation='relu'),
    tf.keras.layers.Dense(units=64, activation='relu'),
    tf.keras.layers.Dense(units=1)
])

history = dense.compile_and_fit( single_step_window)

val_performance['Dense'] = dense.evaluate(single_step_window.val)
performance['Dense'] = dense.evaluate(single_step_window.test, verbose=0)

plt.bar(x = range(len(train_df.columns)),
        height=dense.layers[0].kernel[:,0].numpy())
axis = plt.gca()
axis.set_xticks(range(len(train_df.columns)))
_ = axis.set_xticklabels(train_df.columns, rotation=90)
plt.show()


#////////////////////////#
#          RNN           #
#////////////////////////#



lstm_model = Sequential([
    # Shape [batch, time, features] => [batch, time, lstm_units]
    tf.keras.layers.LSTM(32, return_sequences=True,dropout=0.2),
    # Shape => [batch, time, features]
    tf.keras.layers.Dense(units=1)
],ID='LSTM')


history = lstm_model.compile_and_fit( wide_window,MAX_EPOCHS=30)

IPython.display.clear_output()
val_performance['LSTM'] = lstm_model.evaluate(wide_window.val)
performance['LSTM'] = lstm_model.evaluate(wide_window.test, verbose=0)

wide_window.plot(lstm_model)




# ///////////////////
# conv
# //////////////////


CONV_WIDTH = 3
LABEL_WIDTH = 23
INPUT_WIDTH = LABEL_WIDTH + (CONV_WIDTH - 1)
conv_window = WindowGenerator(
    input_width=INPUT_WIDTH,
    label_width=LABEL_WIDTH,
    shift=1,
    label_columns=[plot_col])


conv_model = Sequential([
    tf.keras.layers.Conv1D(filters=32,
                           kernel_size=(CONV_WIDTH,),
                           activation='relu'),
    tf.keras.layers.Dense(units=32, activation='relu'),
    tf.keras.layers.Dense(units=1),
],ID = 'Conv')


print("Conv model on `conv_window`")
print('Input shape:', conv_window.example[0].shape)
print('Labels shape:', conv_window.example[1].shape)
print('Output shape:', conv_model(conv_window.example[0]).shape)
print(conv_window)

history = conv_model.compile_and_fit(conv_window)

IPython.display.clear_output()
val_performance['Conv'] = conv_model.evaluate(conv_window.val)
performance['Conv'] = conv_model.evaluate(conv_window.test, verbose=0)


conv_window.plot(conv_model)



# ///////////////////
# residual RNN
# //////////////////

class ResidualWrapper(tf.keras.Model):
    def __init__(self, model,ID = 'RESNET'):
        super().__init__()
        self.model = model
        self.id = ID

    def call(self, inputs, *args, **kwargs):
        delta = self.model(inputs, *args, **kwargs)

        # The prediction for each timestep is the input
        # from the previous time step plus the delta
        # calculated by the model.
        return inputs + delta

    def compile_and_fit(self, window, patience=2,MAX_EPOCHS = 100):
        early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                            patience=patience,
                                                            mode='min')

        self.compile(loss=tf.losses.MeanSquaredError(),
                        optimizer=tf.optimizers.Adam(),
                        metrics=[tf.metrics.MeanAbsoluteError()])

        history = self.fit(window.train, epochs=MAX_EPOCHS,
                            validation_data=window.val,
                            callbacks=[early_stopping])
        return history


residual_lstm = ResidualWrapper(
    Sequential([
    tf.keras.layers.LSTM(32, return_sequences=True,dropout=0.2),
    tf.keras.layers.Dense(
        num_features,
        # The predicted deltas should start small
        # So initialize the output layer with zeros
        kernel_initializer=tf.initializers.zeros)
],ID = 'RES'))

history = compile_and_fit(residual_lstm, wide_window)

IPython.display.clear_output()
val_performance['Residual LSTM'] = residual_lstm.evaluate(wide_window.val)
performance['Residual LSTM'] = residual_lstm.evaluate(wide_window.test, verbose=0)
print()


# //////////////
# PERFORMANCE
#///////////////

x = np.arange(len(performance))
width = 0.3
metric_name = 'mean_absolute_error'
metric_index = lstm_model.metrics_names.index('mean_absolute_error')
val_mae = [v[metric_index] for v in val_performance.values()]
test_mae = [v[metric_index] for v in performance.values()]

plt.ylabel('mean_absolute_error [{}], normalized]'.format(plot_col))
plt.bar(x - 0.17, val_mae, width, label='Validation')
plt.bar(x + 0.17, test_mae, width, label='Test')
plt.xticks(ticks=x, labels=performance.keys(),
           rotation=45)
_ = plt.legend()
plt.show()

for name, value in performance.items():
  print(f'{name:12s}: {value[1]:0.4f}')

# ___________________________________________________________
# ___________________________________________________________
# MULTI (single shot)
# -----------------------------------------------------------
# -----------------------------------------------------------


OUT_STEPS = 24
multi_window = WindowGenerator(input_width=24,
                               label_width=OUT_STEPS,
                               shift=OUT_STEPS)

multi_window.plot()
print(multi_window)



# //////////////
# BASELINE
#///////////////

class MultiStepLastBaseline(tf.keras.Model):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.id = 'Baseline'
    def call(self, inputs):
        return tf.tile(inputs[:, -1:, :], [1, OUT_STEPS, 1])

last_baseline = MultiStepLastBaseline()
last_baseline.compile(loss=tf.losses.MeanSquaredError(),
                      metrics=[tf.metrics.MeanAbsoluteError()])

multi_val_performance = {}
multi_performance = {}

multi_val_performance['BENCHMARK'] = last_baseline.evaluate(multi_window.val)
multi_performance['BENCHMARK'] = last_baseline.evaluate(multi_window.test, verbose=0)
multi_window.plot(last_baseline)

print(multi_performance['BENCHMARK'])

# //////////////
# DENSE(Multi)
#///////////////

multi_linear_model = Sequential([
    # Take the last time-step.
    # Shape [batch, time, features] => [batch, 1, features]
    tf.keras.layers.Lambda(lambda x: x[:, -1:, :]),
    # Shape => [batch, 1, out_steps*features]
    tf.keras.layers.Dense(OUT_STEPS*num_features,
                          kernel_initializer=tf.initializers.zeros),
    # Shape => [batch, out_steps, features]
    tf.keras.layers.Reshape([OUT_STEPS, num_features])
],ID = 'Dense(multi)')

history = multi_linear_model.compile_and_fit( multi_window)

IPython.display.clear_output()
multi_val_performance['Linear'] = multi_linear_model.evaluate(multi_window.val)
multi_performance['Linear'] = multi_linear_model.evaluate(multi_window.test, verbose=0)
multi_window.plot(multi_linear_model)

# //////////////
# DENSE(Multi) (RES)
#///////////////

multi_linear_model_RES =ResidualWrapper( Sequential([
    # Take the last time-step.
    # Shape [batch, time, features] => [batch, 1, features]
    tf.keras.layers.Lambda(lambda x: x[:, -1:, :]),
    # Shape => [batch, 1, out_steps*features]
    tf.keras.layers.Dense(OUT_STEPS*num_features,
                          kernel_initializer=tf.initializers.zeros),
    # Shape => [batch, out_steps, features]
    tf.keras.layers.Reshape([OUT_STEPS, num_features])
],ID = 'Dense(multi)'),ID = "Dense(multi_RES)")

history = multi_linear_model_RES.compile_and_fit( multi_window)

IPython.display.clear_output()
multi_val_performance['Linear'] = multi_linear_model_RES.evaluate(multi_window.val)
multi_performance['Linear'] = multi_linear_model_RES.evaluate(multi_window.test, verbose=0)
multi_window.plot(multi_linear_model_RES)
# //////////////
# CONV(Multi)
#///////////////



CONV_WIDTH = 3
multi_conv_model = Sequential([
    # Shape [batch, time, features] => [batch, CONV_WIDTH, features]
    tf.keras.layers.Lambda(lambda x: x[:, -CONV_WIDTH:, :]),
    # Shape => [batch, 1, conv_units]
    tf.keras.layers.Conv1D(256, activation='relu', kernel_size=(CONV_WIDTH)),
    # Shape => [batch, 1,  out_steps*features]
    tf.keras.layers.Dense(OUT_STEPS*num_features,
                          kernel_initializer=tf.initializers.zeros),
    # Shape => [batch, out_steps, features]
    tf.keras.layers.Reshape([OUT_STEPS, num_features])
],ID='conv')

history = multi_conv_model.compile_and_fit( multi_window)

IPython.display.clear_output()

multi_val_performance['Conv'] = multi_conv_model.evaluate(multi_window.val)
multi_performance['Conv'] = multi_conv_model.evaluate(multi_window.test, verbose=0)
multi_window.plot(multi_conv_model)

# //////////////
# CONV(Multi) with RES
#///////////////

CONV_WIDTH = 3
multi_conv_model_res = ResidualWrapper(Sequential([
    # Shape [batch, time, features] => [batch, CONV_WIDTH, features]
    tf.keras.layers.Lambda(lambda x: x[:, -CONV_WIDTH:, :]),
    # Shape => [batch, 1, conv_units]
    tf.keras.layers.Conv1D(256, activation='relu', kernel_size=(CONV_WIDTH)),
    # Shape => [batch, 1,  out_steps*features]
    tf.keras.layers.Dense(OUT_STEPS*num_features,
                          kernel_initializer=tf.initializers.zeros),
    # Shape => [batch, out_steps, features]
    tf.keras.layers.Reshape([OUT_STEPS, num_features])
],ID='conv_RES'),ID = 'conv_RES')

history = multi_conv_model_res.compile_and_fit( multi_window)

IPython.display.clear_output()

multi_val_performance['Conv_RES'] = multi_conv_model_res.evaluate(multi_window.val)
multi_performance['Conv_RES'] = multi_conv_model_res.evaluate(multi_window.test, verbose=0)
multi_window.plot(multi_conv_model_res)

# //////////////
# LSTM(Multi)
#///////////////


multi_lstm_model = Sequential([
    # Shape [batch, time, features] => [batch, lstm_units]
    # Adding more `lstm_units` just overfits more quickly.
    tf.keras.layers.LSTM(32, return_sequences=False,dropout=0.2),
    # Shape => [batch, out_steps*features]
    tf.keras.layers.Dense(OUT_STEPS*num_features,
                          kernel_initializer=tf.initializers.zeros),
    # Shape => [batch, out_steps, features]
    tf.keras.layers.Reshape([OUT_STEPS, num_features])
],ID = 'LSTM')

history = compile_and_fit(multi_lstm_model, multi_window)

IPython.display.clear_output()

multi_val_performance['LSTM'] = multi_lstm_model.evaluate(multi_window.val)
multi_performance['LSTM'] = multi_lstm_model.evaluate(multi_window.test, verbose=0)
multi_window.plot(multi_lstm_model)



# //////////////
# LSTM(residual)
#///////////////

class FeedBack(tf.keras.Model):
    def __init__(self, units, out_steps):
        super().__init__()
        self.out_steps = out_steps
        self.units = units
        self.lstm_cell = tf.keras.layers.LSTMCell(units,dropout = 0.1)
        # Also wrap the LSTMCell in an RNN to simplify the `warmup` method.
        self.lstm_rnn = tf.keras.layers.RNN(self.lstm_cell, return_state=True)
        self.dense = tf.keras.layers.Dense(num_features)
        self.id = 'AUTOREG'

    def warmup(self, inputs):
        # inputs.shape => (batch, time, features)
        # x.shape => (batch, lstm_units)
        x, *state = self.lstm_rnn(inputs)

        # predictions.shape => (batch, features)
        prediction = self.dense(x)
        return prediction, state
    def call(self, inputs, training=None):
        # Use a TensorArray to capture dynamically unrolled outputs.
        predictions = []
        # Initialize the lstm state
        prediction, state = self.warmup(inputs)

        # Insert the first prediction
        predictions.append(prediction)

        # Run the rest of the prediction steps
        for n in range(1, self.out_steps):
            # Use the last prediction as input.
            x = prediction
            # Execute one lstm step.
            x, state = self.lstm_cell(x, states=state,
                                    training=training)
            # Convert the lstm output to a prediction.
            prediction = self.dense(x)
            # Add the prediction to the output
            predictions.append(prediction)

        # predictions.shape => (time, batch, features)
        predictions = tf.stack(predictions)
        # predictions.shape => (batch, time, features)
        predictions = tf.transpose(predictions, [1, 0, 2])
        return predictions


feedback_model =  ResidualWrapper(
    FeedBack(units=32, out_steps=OUT_STEPS),
    ID = "FEEBACK RES")



history = compile_and_fit(feedback_model, multi_window)

IPython.display.clear_output()

multi_val_performance['AR RES LSTM'] = feedback_model.evaluate(multi_window.val)
multi_performance['AR RES LSTM'] = feedback_model.evaluate(multi_window.test, verbose=0)
print(feedback_model.evaluate(multi_window.test, verbose=0))



# //////////////
# PERFORMANCE
#///////////////


for name, value in multi_performance.items():
  print(f'{name}: {value[1]}')

x = np.arange(len(multi_performance))
width = 0.3


metric_name = 'mean_absolute_error'
metric_index = lstm_model.metrics_names.index('mean_absolute_error')
val_mae = [v[metric_index] for v in multi_val_performance.values()]
test_mae = [v[metric_index] for v in multi_performance.values()]

plt.bar(x - 0.17, val_mae, width, label='Validation')
plt.bar(x + 0.17, test_mae, width, label='Test')
plt.xticks(ticks=x, labels=multi_performance.keys(),
           rotation=45)
plt.ylabel(f'MAE (average over all times and outputs)')
_ = plt.legend()
plt.show()




