import parl
from parl import layers  # 封装了 paddle.fluid.layers 的API

class Model(parl.Model):
    def __init__(self, act_dim):
        act_dim = act_dim
        hidden_dim_1 = hidden_dim_2 = 128

        self.fc1 = layers.fc(size=hidden_dim_1, act='tanh')
        self.fc2 = layers.fc(size=hidden_dim_2, act='tanh')
        self.fc3 = layers.fc(size=act_dim, act="softmax")

    def forward(self, obs):  # 可直接用 model = Model(5); model(obs)调用
        out = self.fc1(obs)
        out = self.fc2(out)
        out = self.fc3(out)
        return out