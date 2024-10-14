import numpy as np
from keras.src import backend
from keras.src.layers import LSTM, Conv1D, Dense, Flatten, Input, MaxPooling1D
from keras.src.models import Model


class SharedModel:
    def __init__(self, input_shape, action_space, learning_rate, optimizer, model='Dense'):
        x_input = Input(input_shape)
        self.action_space = action_space

        # Shared CNN layers:
        if model == 'CNN':
            x = Conv1D(filters=64, kernel_size=6, padding='same', activation='tanh')(
                x_input
            )
            x = MaxPooling1D(pool_size=2)(x)
            x = Conv1D(filters=32, kernel_size=3, padding='same', activation='tanh')(x)
            x = MaxPooling1D(pool_size=2)(x)
            x = Flatten()(x)

        # Shared LSTM layers:
        elif model == 'LSTM':
            x = LSTM(512, return_sequences=True)(x_input)
            x = LSTM(256)(x)

        # Shared Dense layers:
        else:
            x = Flatten()(x_input)
            x = Dense(512, activation='relu')(x)

        # Critic model
        v = Dense(512, activation='relu')(x)
        v = Dense(256, activation='relu')(v)
        v = Dense(64, activation='relu')(v)
        value = Dense(1, activation=None)(v)

        self.Critic = Model(inputs=x_input, outputs=value)
        self.Critic.compile(
            loss=self.critic_ppo2_loss, optimizer=optimizer(learning_rate=learning_rate)
        )

        # Actor model
        a = Dense(512, activation='relu')(x)
        a = Dense(256, activation='relu')(a)
        a = Dense(64, activation='relu')(a)
        output = Dense(self.action_space, activation='softmax')(a)

        self.Actor = Model(inputs=x_input, outputs=output)
        self.Actor.compile(loss=self.ppo_loss, optimizer=optimizer(learning_rate=learning_rate))
        # print(self.Actor.summary())

    def ppo_loss(self, y_true, y_pred):
        # Defined in https://arxiv.org/abs/1707.06347
        advantages, prediction_picks, actions = (
            y_true[:, :1],
            y_true[:, 1 : 1 + self.action_space],
            y_true[:, 1 + self.action_space :],
        )
        loss_clipping = 0.2
        entropy_loss = 0.001

        prob = actions * y_pred
        old_prob = actions * prediction_picks

        prob = backend.clip(prob, 1e-10, 1.0)
        old_prob = backend.clip(old_prob, 1e-10, 1.0)

        ratio = backend.exp(backend.log(prob) - backend.log(old_prob))

        p1 = ratio * advantages
        p2 = (
            backend.clip(
                ratio, min_value=1 - loss_clipping, max_value=1 + loss_clipping
            )
            * advantages
        )

        actor_loss = -backend.mean(backend.minimum(p1, p2))

        entropy = -(y_pred * backend.log(y_pred + 1e-10))
        entropy = entropy_loss * backend.mean(entropy)

        total_loss = actor_loss - entropy

        return total_loss

    def actor_predict(self, state):
        return self.Actor.predict(state)

    def critic_ppo2_loss(self, y_true, y_pred):
        value_loss = backend.mean((y_true - y_pred) ** 2)  # standard PPO loss
        return value_loss

    def critic_predict(self, state):
        return self.Critic.predict([state, np.zeros((state.shape[0], 1))])


class ActorModel:
    def __init__(self, input_shape, action_space, learning_rate, optimizer):
        x_input = Input(input_shape)
        self.action_space = action_space

        x = Flatten(input_shape=input_shape)(x_input)
        x = Dense(512, activation='relu')(x)
        x = Dense(256, activation='relu')(x)
        x = Dense(64, activation='relu')(x)
        output = Dense(self.action_space, activation='softmax')(x)

        self.Actor = Model(inputs=x_input, outputs=output)
        self.Actor.compile(loss=self.ppo_loss, optimizer=optimizer(learning_rate=learning_rate))
        # print(self.Actor.summary)

    def ppo_loss(self, y_true, y_pred):
        # Defined in https://arxiv.org/abs/1707.06347
        advantages, prediction_picks, actions = (
            y_true[:, :1],
            y_true[:, 1 : 1 + self.action_space],
            y_true[:, 1 + self.action_space :],
        )
        loss_clipping = 0.2
        entropy_loss = 0.001

        prob = actions * y_pred
        old_prob = actions * prediction_picks

        prob = backend.clip(prob, 1e-10, 1.0)
        old_prob = backend.clip(old_prob, 1e-10, 1.0)

        ratio = backend.exp(backend.log(prob) - backend.log(old_prob))

        p1 = ratio * advantages
        p2 = (
            backend.clip(
                ratio, min_value=1 - loss_clipping, max_value=1 + loss_clipping
            )
            * advantages
        )

        actor_loss = -backend.mean(backend.minimum(p1, p2))

        entropy = -(y_pred * backend.log(y_pred + 1e-10))
        entropy = entropy_loss * backend.mean(entropy)

        total_loss = actor_loss - entropy

        return total_loss

    def actor_predict(self, state):
        return self.Actor.predict(state)


class CriticModel:
    def __init__(self, input_shape, action_space, learning_rate, optimizer):
        x_input = Input(input_shape)

        v = Flatten(input_shape=input_shape)(x_input)
        v = Dense(512, activation='relu')(v)
        v = Dense(256, activation='relu')(v)
        v = Dense(64, activation='relu')(v)
        value = Dense(1, activation=None)(v)

        self.Critic = Model(inputs=x_input, outputs=value)
        self.Critic.compile(
            loss=self.critic_ppo2_loss, optimizer=optimizer(learning_rate=learning_rate)
        )

    def critic_ppo2_loss(self, y_true, y_pred):
        value_loss = backend.mean((y_true - y_pred) ** 2)  # standard PPO loss
        return value_loss

    def critic_predict(self, state):
        return self.Critic.predict([state, np.zeros((state.shape[0], 1))])
