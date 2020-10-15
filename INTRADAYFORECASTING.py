import numpy as np
import pandas as pd
from datetime import datetime
import pandas_datareader as dr 
import pandas_datareader.data as web
from matplotlib import pyplot as plt
import seaborn as sns
import math


import plotly.graph_objects as go
import plotly.express as px

import tensorflow as tf

import IPython
import IPython.display

#/////////////////////////#
#        Formatting       #
#/////////////////////////#


sns.set(style='darkgrid', context='paper', palette='Dark2')
# sns.set(rc={'axes.facecolor':'#1D2128'})
sns.despine()





#//////////////////////////////#
#     Load &  preprocess       #
#//////////////////////////////#

# STOCKS
DATA = [pd.read_csv("STOCK_DATA/NDAQ_STOCK_DATA_ADJUSTED_1min_INTERVALS_{}.csv".format(i))for i in range(1,11)]
STOCK_DATA = pd.concat(DATA).drop_duplicates()


STOCK_DATA.drop(columns  = ["Unnamed: 0"],inplace = True)
STOCK_DATA.drop(columns  = ['volume'],inplace = True)
STOCK_DATA.sort_values(by=['time'],ascending = True,inplace = True)
STOCK_DATA.reset_index(drop=True,inplace = True)
STOCK_DATA['Rolling_Hourly'] = STOCK_DATA['close'].rolling(window=20).mean()
STOCK_DATA.dropna(inplace = True)
# MSFT_DATA = MSFT_DATA[::20]
STOCK_DATA.time = STOCK_DATA.time.apply(lambda d: datetime.strptime(d, '%Y-%m-%d %H:%M:%S').date())
STOCK_DATA = STOCK_DATA[STOCK_DATA.time <=datetime(2020,10,3).date()]
TIMESTAMPS = STOCK_DATA['time']





# COVID-US
COVID_DATA = pd.read_csv("COVID_DATA/us-counties-NYT.csv")
COVID_DATA = COVID_DATA.drop(columns=['county','state','fips']).groupby(['date']).sum()
POLICY_DATA = pd.read_csv("COVID_DATA/covid_policies_engineered.csv")
POLICY_DATA.set_index('date_announced',inplace = True)

print(POLICY_DATA.head())
print(COVID_DATA.describe())

# print(POLICY_DATA['date_announced'].dt)

def get_cases(date):
    try:
        return COVID_DATA.loc[date.strftime("%Y-%m-%d")]
    except:
        return COVID_DATA.loc[datetime(2020,10,3).strftime("%Y-%m-%d")]

APPENDAGE = TIMESTAMPS.apply(get_cases)
# STOCK_DATA['Cases'] = APPENDAGE['cases']
# STOCK_DATA['Deaths'] = APPENDAGE['deaths']


def get_cell(date):
    try:
        
        return POLICY_DATA.loc[date.strftime("%Y-%m-%d")]
        
    except:
        
        return POLICY_DATA.loc[datetime(2020,8,25).strftime("%Y-%m-%d")]

       
        


APPENDAGE2 = TIMESTAMPS.apply(get_cell)
for Col in ["entry_type", "correct_type", "update_type",  "update_level",  "init_country_level",   "compliance",'Border','Regulation','Lockdown','Curfew','Social Distancing']:
    STOCK_DATA[Col] = APPENDAGE2[Col]





STOCK_DATA.fillna(0,inplace = True)
STOCK_DATA.drop(columns = ['time'],inplace = True)

print(POLICY_DATA.describe())

print(STOCK_DATA.head())


print(STOCK_DATA.describe())
correlation = STOCK_DATA.corr()
Correlation_plot = px.imshow(correlation)
Correlation_plot.show()

#///////////////////////////////#
#        train test split       #
#///////////////////////////////#

column_indices = {name: i for i, name in enumerate(STOCK_DATA.columns)}

n = len(STOCK_DATA)
train_df = STOCK_DATA[0:int(n*0.55)]
val_df = STOCK_DATA[int(n*0.55):int(n*0.8)]
test_df= STOCK_DATA[int(n*0.8):]

num_features = STOCK_DATA.shape[1]
print("NUM FEATURES: {}".format(num_features))

# Normalise data

train_mean = train_df.mean()
train_std = train_df.std()


train_df = (train_df - train_mean) / train_std
val_df = (val_df - train_mean) / train_std
test_df = (test_df - train_mean) / train_std

df_std = (STOCK_DATA - train_mean) / train_std
df_std = df_std.melt(var_name='Column', value_name='Normalized')
plt.figure(figsize=(12, 6))
ax = sns.violinplot(x='Column', y='Normalized', data=df_std)
_ = ax.set_xticklabels(STOCK_DATA.keys(), rotation=90)

plt.show()



plot_col = 'Rolling_Hourly'


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
                    label='Inputs', zorder=-10,c="#1aa8b4")

            if self.label_columns:
                label_col_index = self.label_columns_indices.get(plot_col, None)
            else:
                label_col_index = plot_col_index

            if label_col_index is None:
                continue


            plt.plot(self.label_indices, labels[n, :, label_col_index],
                        label='Labels', zorder=-10,c="#1aa8b4")
            if model is not None:
                predictions = model(inputs)
                plt.plot(self.label_indices, predictions[n, :, label_col_index],
                 label='Predictions',c = "#FF1493")

            if n == 0:
                plt.legend()

            plt.xlabel('Timestep')

            if model and hasattr(model,'id'):
                plt.title(model.id)

        
            
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



# multi_window
OUT_STEPS = 392
multi_window = WindowGenerator(input_width=392,
                               label_width=OUT_STEPS,
                               shift=OUT_STEPS)



CONV_WIDTH = 3
LABEL_WIDTH = 392
INPUT_WIDTH = LABEL_WIDTH + (CONV_WIDTH - 1)
conv_window = WindowGenerator(
    input_width=INPUT_WIDTH,
    label_width=LABEL_WIDTH,
    shift=1,
    label_columns=[plot_col])



# ___________________________________________________________
# ___________________________________________________________
# MODELS
# -----------------------------------------------------------
# -----------------------------------------------------------

#////////////////////////#
#         Learning rate      #
#////////////////////////#

lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate = 0.01,
    decay_steps=100000,
    decay_rate=0.90,
    staircase=True)

# plt.plot(lr_schedule)
# plt.ylabel("Learning Rate")
# plt.xlabel("Train Step")
# plt.show()

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
        self.checkpoint_filepath = './checkpoint/{0}/{0}'.format(self.id)
        

    def compile_and_fit(self, window, patience=10,MAX_EPOCHS = 200,baseline = None):

        model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
                filepath=self.checkpoint_filepath,
                save_weights_only=True,
                monitor='val_loss',
                mode='max',
                save_best_only=True)

        early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                            patience=patience,
                                                            min_delta=0.0001,
                                                            mode='min',

                                                            baseline = baseline,
                                                            restore_best_weights=True)

        self.compile(loss=tf.losses.MeanSquaredError(),
                        optimizer=tf.optimizers.Adam(learning_rate=lr_schedule),
                        metrics=[tf.metrics.MeanAbsoluteError()])

        history = self.fit(window.train, epochs=MAX_EPOCHS,
                            validation_data=window.val,
                            callbacks=[early_stopping,model_checkpoint_callback])
        return history
    
    def load(self):
        self.load_weights(self.checkpoint_filepath)




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


    
    



class Evaluator():
    def __init__(self):
        self.validation_performance = {}
        self.test_performance = {}
        

    def evaluate(self,model,window,name):
        self.validation_performance[name] = model.evaluate(window.val)
        self.test_performance[name] = model.evaluate(window.test)

    def performance(self,model):
        x = np.arange(len(self.test_performance))
        width = 0.3
        metric_name = 'mean_absolute_error'
        metric_index =  metric_index = model.metrics_names.index('mean_absolute_error')
        val_mae = [v[metric_index] for v in self.validation_performance.values()]
        test_mae = [v[metric_index] for v in self.test_performance.values()]

        plt.ylabel('mean_absolute_error [{}], normalized]'.format(plot_col))
        plt.bar(x - 0.17, val_mae, width, label='Validation')
        plt.bar(x + 0.17, test_mae, width, label='Test')
        plt.xticks(ticks=x, labels=self.test_performance.keys(),
                rotation=45)
        _ = plt.legend()
        plt.show()

        for name, value in self.test_performance.items():
            print(f'{name:12s}: {value[1]:0.4f}')


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

class ResidualWrapper(tf.keras.Model):
    def __init__(self, model,ID = 'RESNET'):
        super().__init__()
        self.model = model
        self.id = ID
        self.checkpoint_filepath = './checkpoint/{0}/{0}'.format(self.id)

    def call(self, inputs, *args, **kwargs):
        delta = self.model(inputs, *args, **kwargs)

        # The prediction for each timestep is the input
        # from the previous time step plus the delta
        # calculated by the model.
        return inputs + delta



    def compile_and_fit(self, window, patience=10,MAX_EPOCHS = 200,baseline = None):

        model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
                filepath=self.checkpoint_filepath,
                save_weights_only=True,
                monitor='val_loss',
                mode='max',
                save_best_only=True)

        early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                            patience=patience,
                                                            min_delta=0.0001,
                                                            mode='min',

                                                            baseline = baseline,
                                                            restore_best_weights=True)

        self.compile(loss=tf.losses.MeanSquaredError(),
                        optimizer=tf.optimizers.Adam(learning_rate=lr_schedule),
                        metrics=[tf.metrics.MeanAbsoluteError()])

        history = self.fit(window.train, epochs=MAX_EPOCHS,
                            validation_data=window.val,
                            callbacks=[early_stopping,model_checkpoint_callback])
        return history
    
    def load(self):
        self.load_weights(self.checkpoint_filepath)



lstm_model = Sequential([
        # Shape [batch, time, features] => [batch, time, lstm_units]
        tf.keras.layers.LSTM(32, return_sequences=True,dropout=0.2),
        # Shape => [batch, time, features]
        tf.keras.layers.Dense(units=1)
    ],ID='LSTM')


history = lstm_model.compile_and_fit( wide_window,MAX_EPOCHS=1)


single = False

if single:

    SS_Evaluator = Evaluator()

    #////////////////////////#
    #         Baseline       #
    #////////////////////////#


    baseline = Baseline(label_index=column_indices[plot_col],ID = 'Baseline')

    baseline.compile(loss=tf.losses.MeanSquaredError(),
                    metrics=[tf.metrics.MeanAbsoluteError()])

    SS_Evaluator.evaluate(baseline,single_step_window,'Baseline')



    #////////////////////////#
    #         linear         #
    #////////////////////////#

    linear = Sequential([
        tf.keras.layers.Dense(units=1)
    ])


    history = linear.compile_and_fit( single_step_window)

    linear.load()

    SS_Evaluator.evaluate(linear,single_step_window,'Linear')




    # #////////////////////////#
    # #         multidense     #
    # #////////////////////////#

    # dense = Sequential([
    #     tf.keras.layers.Dense(units=64, activation='relu'),
    #     tf.keras.layers.Dense(units=64, activation='relu'),
    #     tf.keras.layers.Dense(units=1)
    # ])

    # history = dense.compile_and_fit( single_step_window)

    # SS_Evaluator.evaluate(dense,single_step_window,'Dense')


    #////////////////////////#
    #          RNN           #
    #////////////////////////#



    lstm_model = Sequential([
        # Shape [batch, time, features] => [batch, time, lstm_units]
        tf.keras.layers.LSTM(32, return_sequences=True,dropout=0.2),
        # Shape => [batch, time, features]
        tf.keras.layers.Dense(units=1)
    ],ID='LSTM')


    history = lstm_model.compile_and_fit( wide_window,MAX_EPOCHS=10)

    lstm_model.load()

    IPython.display.clear_output()
    SS_Evaluator.evaluate(lstm_model,wide_window,'LSTM')






    # ///////////////////
    # conv
    # //////////////////





    conv_model = Sequential([
        tf.keras.layers.Conv1D(filters=32,
                            kernel_size=(CONV_WIDTH,),
                            activation='relu'),
        tf.keras.layers.Dense(units=32, activation='relu'),
        tf.keras.layers.Dense(units=1),
    ],ID = 'Conv')




    history = conv_model.compile_and_fit(conv_window)
    conv_model.load()

    IPython.display.clear_output()
    SS_Evaluator.evaluate(conv_model,conv_window,"CNN")


    conv_window.plot(conv_model)



    # ///////////////////
    # residual RNN
    # //////////////////




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
    SS_Evaluator.evaluate(residual_lstm,wide_window,"RES LSTM")


    # //////////////
    # PERFORMANCE
    #///////////////

    SS_Evaluator.performance(lstm_model)




# ___________________________________________________________
# ___________________________________________________________
# MULTI (single shot)
# -----------------------------------------------------------
# -----------------------------------------------------------


MS_Evaluator = Evaluator()

class MultiStepLastBaseline(tf.keras.Model):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.id = 'Baseline'
    def call(self, inputs):
        return tf.tile(inputs[:, -1:, :], [1, OUT_STEPS, 1])

last_baseline = MultiStepLastBaseline()
last_baseline.compile(loss=tf.losses.MeanSquaredError(),
                      metrics=[tf.metrics.MeanAbsoluteError()])


MS_Evaluator.evaluate(last_baseline,multi_window,'BASELINE')

multi_window.plot(last_baseline)


# //////////////
# DENSE(Multi)
#///////////////

# multi_linear_model_basic = Sequential([
#     # Take the last time-step.
#     # Shape [batch, time, features] => [batch, 1, features]
#     tf.keras.layers.Lambda(lambda x: x[:, -1:, :]),
#     # Shape => [batch, 1, out_steps*features]
#     tf.keras.layers.Dense(128,
#                           kernel_initializer=tf.initializers.zeros,activation = 'relu'),
#     # Shape => [batch, out_steps, features]
#     tf.keras.layers.Reshape([OUT_STEPS, num_features])
# ],ID = 'Dense_basic')

# history = multi_linear_model_basic.compile_and_fit( multi_window)

# IPython.display.clear_output()
# MS_Evaluator.evaluate(multi_linear_model_basic,multi_window,'DENSE')

# multi_window.plot(multi_linear_model_basic)



multi_linear_model = Sequential([
    # Take the last time-step.
    # Shape [batch, time, features] => [batch, 1, features]
    tf.keras.layers.Lambda(lambda x: x[:, -1:, :]),
    # Shape => [batch, 1, out_steps*features]
    tf.keras.layers.Dense(OUT_STEPS*num_features,
                          kernel_initializer=tf.initializers.zeros),
    # Shape => [batch, out_steps, features]
    tf.keras.layers.Reshape([OUT_STEPS, num_features])
],ID = 'Dense')

history = multi_linear_model.compile_and_fit( multi_window)

IPython.display.clear_output()
MS_Evaluator.evaluate(multi_linear_model,multi_window,'DENSE')

multi_window.plot(multi_linear_model)

multi_linear_model5 = Sequential([
    # Take the last time step.
    # Shape [batch, time, features] => [batch, 1, features]
    tf.keras.layers.Lambda(lambda x: x[:, -1:, :]),
    # Shape => [batch, 1, dense_units]
    tf.keras.layers.Dense(16, activation='relu'),
    # Shape => [batch, out_steps*features]
    tf.keras.layers.Dense(OUT_STEPS*num_features,
                          kernel_initializer=tf.initializers.zeros),
    # Shape => [batch, out_steps, features]
    tf.keras.layers.Reshape([OUT_STEPS, num_features])
],ID = 'Dense16')

IPython.display.clear_output()
history = multi_linear_model5.compile_and_fit( multi_window)
MS_Evaluator.evaluate(multi_linear_model5,multi_window,'DENSE_16')
multi_window.plot(multi_linear_model5)



# multi_linear_model2 = Sequential([
#     # Take the last time step.
#     # Shape [batch, time, features] => [batch, 1, features]
#     tf.keras.layers.Lambda(lambda x: x[:, -1:, :]),
#     # Shape => [batch, 1, dense_units]
#     tf.keras.layers.Dense(16, activation='relu'),
#     tf.keras.layers.Dense(8, activation='relu'),
#     tf.keras.layers.Dense(16, activation='relu'),
#     # Shape => [batch, out_steps*features]
#     tf.keras.layers.Dense(OUT_STEPS*num_features,
#                           kernel_initializer=tf.initializers.zeros),
#     # Shape => [batch, out_steps, features]
#     tf.keras.layers.Reshape([OUT_STEPS, num_features])
# ],ID = 'Dense2')

# history = multi_linear_model2.compile_and_fit( multi_window)

# IPython.display.clear_output()
# MS_Evaluator.evaluate(multi_linear_model2,multi_window,'DENSE_16')
# multi_window.plot(multi_linear_model2)

# multi_linear_model4 = Sequential([
#     # Take the last time step.
#     # Shape [batch, time, features] => [batch, 1, features]
#     tf.keras.layers.Lambda(lambda x: x[:, -1:, :]),
#     # Shape => [batch, 1, dense_units]
#     tf.keras.layers.Dense(16, activation='relu'),
#     tf.keras.layers.Dense(8, activation='relu'),
#     tf.keras.layers.Dense(8, activation='relu'),
#     tf.keras.layers.Dense(16, activation='relu'),
#     # Shape => [batch, out_steps*features]
#     tf.keras.layers.Dense(OUT_STEPS*num_features,
#                           kernel_initializer=tf.initializers.zeros),
#     # Shape => [batch, out_steps, features]
#     tf.keras.layers.Reshape([OUT_STEPS, num_features])
# ],ID = 'Dense162x8')

# history = multi_linear_model4.compile_and_fit( multi_window)

# IPython.display.clear_output()
# MS_Evaluator.evaluate(multi_linear_model4,multi_window,'DENSE_16_2x8')
# multi_window.plot(multi_linear_model4)


# multi_linear_model3 = Sequential([
#         # Take the last time step.
#     # Shape [batch, time, features] => [batch, 1, features]
#     tf.keras.layers.Lambda(lambda x: x[:, -1:, :]),
#     # Shape => [batch, 1, dense_units]
#     tf.keras.layers.Dense(8, activation='relu'),
#     tf.keras.layers.Dense(8, activation='relu'),
#     tf.keras.layers.Dense(8, activation='relu'),
#     tf.keras.layers.Dense(8, activation='relu'),
#     # Shape => [batch, out_steps*features]
#     tf.keras.layers.Dense(OUT_STEPS*num_features,
#                           kernel_initializer=tf.initializers.zeros),
#     # Shape => [batch, out_steps, features]
#     tf.keras.layers.Reshape([OUT_STEPS, num_features])
# ],ID = 'Dense3')

# history = multi_linear_model3.compile_and_fit( multi_window)

# IPython.display.clear_output()
# MS_Evaluator.evaluate(multi_linear_model3,multi_window,'DENSE_8')
# multi_window.plot(multi_linear_model3)
# # //////////////
# DENSE(Multi) (RES)
#///////////////

# multi_linear_model_RES =ResidualWrapper( Sequential([
#     # Take the last time-step.
#     # Shape [batch, time, features] => [batch, 1, features]
#     tf.keras.layers.Lambda(lambda x: x[:, -1:, :]),
#     # Shape => [batch, 1, out_steps*features]
#     tf.keras.layers.Dense(64,
#                           kernel_initializer=tf.initializers.zeros,activation = "relu"),
#     # Shape => [batch, out_steps, features]
#     tf.keras.layers.Reshape([OUT_STEPS, num_features])
# ],ID = 'Dense(RES)'),ID = "Dense(RES)")

# history = multi_linear_model_RES.compile_and_fit( multi_window)

# IPython.display.clear_output()
# MS_Evaluator.evaluate(multi_linear_model_RES,multi_window,'DENSE (RES)')
# multi_window.plot(multi_linear_model_RES )


# //////////////
# CONV(Multi)
#///////////////



# CONV_WIDTH = 3
# multi_conv_model = Sequential([
#     # Shape [batch, time, features] => [batch, CONV_WIDTH, features]
#     tf.keras.layers.Lambda(lambda x: x[:, -CONV_WIDTH:, :]),
#     # Shape => [batch, 1, conv_units]
#     tf.keras.layers.Conv1D(64, activation='relu', kernel_size=(CONV_WIDTH),padding="VALID"),
    
#     tf.keras.layers.Conv1D(filters=64, kernel_size=(CONV_WIDTH), activation='relu',padding="SAME"),
#     tf.keras.layers.MaxPooling1D(pool_size=2,strides=1,padding="SAME"),
#     tf.keras.layers.Flatten(),

#     tf.keras.layers.Dropout(0.2, noise_shape=None, seed=None,),

#     # Shape => [batch, 1,  out_steps*features]
#     # Shape => [batch, 1,  out_steps*features]
#     tf.keras.layers.Dense(OUT_STEPS*num_features,
#                           kernel_initializer=tf.initializers.zeros,activation = 'softmax'),
#     # Shape => [batch, out_steps, features]
#     tf.keras.layers.Reshape([OUT_STEPS, num_features])
# ],ID='conv')


# history = multi_conv_model.compile_and_fit( conv_window)

# IPython.display.clear_output()
# MS_Evaluator.evaluate(multi_conv_model,conv_window,'CNN')

# conv_window.plot(multi_conv_model )


# //////////////
# CONV(Multi) with RES
#///////////////

# CONV_WIDTH = 3
# multi_conv_model_res = ResidualWrapper(Sequential([
#     # Shape [batch, time, features] => [batch, CONV_WIDTH, features]
#     tf.keras.layers.Lambda(lambda x: x[:, -CONV_WIDTH:, :]),
#     # Shape => [batch, 1, conv_units]
#     tf.keras.layers.Conv1D(256, activation='relu', kernel_size=(CONV_WIDTH)),

#     tf.keras.layers.Dropout(0.2, noise_shape=None, seed=None,),
#     # Shape => [batch, 1,  out_steps*features]
#     tf.keras.layers.Dense(OUT_STEPS*num_features,
#                           kernel_initializer=tf.initializers.zeros),
#     # Shape => [batch, out_steps, features]
#     tf.keras.layers.Reshape([OUT_STEPS, num_features])
# ],ID='conv_RES'),ID = 'conv_RES')

# history = multi_conv_model_res.compile_and_fit( multi_window)

# IPython.display.clear_output()
# MS_Evaluator.evaluate(multi_conv_model_res,multi_window,'CNN_RES')

# multi_window.plot(multi_conv_model_res)




# //////////////
# LSTM(residual)
#///////////////




feedback_model =  ResidualWrapper(
    FeedBack(units=32, out_steps=OUT_STEPS),
    ID = "FEEBACK RES")



history = compile_and_fit(feedback_model, multi_window)

IPython.display.clear_output()
MS_Evaluator.evaluate(feedback_model,multi_window,'FEEDBACK LSTM')

multi_window.plot(feedback_model)

MS_Evaluator.performance(lstm_model)