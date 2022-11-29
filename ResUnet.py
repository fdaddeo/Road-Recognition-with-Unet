from keras.layers import Input, Conv2D, Activation, Add, UpSampling2D, Concatenate
from keras.models import Model
from tensorflow_addons.layers import InstanceNormalization

class ResUnet:
    def __init__(self, image_size, image_channels):
        self._f = [16, 32, 64, 128, 256]
        self._inputs = Input((image_size, image_size, image_channels))

    def build_net(self):
      ## Encoder
      e0 = self._inputs
      e1 = self.stem(e0, self._f[0])
      e2 = self.residual_block(e1, self._f[1], strides=2)
      e3 = self.residual_block(e2, self._f[2], strides=2)
      e4 = self.residual_block(e3, self._f[3], strides=2)
      e5 = self.residual_block(e4, self._f[4], strides=2)
      
      ## Bridge
      b0 = self.conv_block(e5, self._f[4], strides=1)
      b1 = self.conv_block(b0, self._f[4], strides=1)
      
      ## Decoder
      u1 = self.upsample_concat_block(b1, e4)
      d1 = self.residual_block(u1, self._f[4])
      
      u2 = self.upsample_concat_block(d1, e3)
      d2 = self.residual_block(u2, self._f[3])
      
      u3 = self.upsample_concat_block(d2, e2)
      d3 = self.residual_block(u3, self._f[2])
      
      u4 = self.upsample_concat_block(d3, e1)
      d4 = self.residual_block(u4, self._f[1])
      
      outputs = Conv2D(1, (1, 1), padding="same", activation="sigmoid")(d4)
      model = Model(self._inputs, outputs)

      return model

    def conv_block(self, x, filters, kernel_size=(3, 3), padding="same", strides=1):
        conv = InstanceNormalization()(x)
        conv = Activation("relu")(conv)
        conv = Conv2D(filters, kernel_size, padding=padding, strides=strides)(conv)

        return conv

    def stem(self, x, filters, kernel_size=(3, 3), padding="same", strides=1):
        conv = Conv2D(filters, kernel_size, padding=padding, strides=strides)(x)
        conv = self.conv_block(conv, filters, kernel_size=kernel_size, padding=padding, strides=strides)
        
        shortcut = Conv2D(filters, kernel_size=(1, 1), padding=padding, strides=strides)(x)
        shortcut = InstanceNormalization()(shortcut)
        
        output = Add()([conv, shortcut])

        return output

    def residual_block(self, x, filters, kernel_size=(3, 3), padding="same", strides=1):
        res = self.conv_block(x, filters, kernel_size=kernel_size, padding=padding, strides=strides)
        res = self.conv_block(res, filters, kernel_size=kernel_size, padding=padding, strides=1)
        
        shortcut = Conv2D(filters, kernel_size=(1, 1), padding=padding, strides=strides)(x)
        shortcut = InstanceNormalization()(shortcut)
        
        output = Add()([shortcut, res])

        return output

    def upsample_concat_block(self, x, xskip):
        u = UpSampling2D((2, 2))(x)
        c = Concatenate()([u, xskip])

        return c