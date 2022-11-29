from keras.layers import Conv2D, BatchNormalization, Activation, MaxPool2D, Conv2DTranspose, Concatenate, Input
from keras.models import Model
from numpy import dtype
from tensorflow_addons.layers import InstanceNormalization

# Conv2D, Conv2DTranspose:
# num_filters = number of output filters in the convolution
# kernel_size = 3x3
# strides =	int/ints specifying the strides of the convolution along the height and width
# padding same = padding with zeros evenly such that output has the same height/width dimension as the input

# BatchNormalization:
# applies a transformation that maintains the mean output close to 0
# and the output standard deviation close to 1

# MaxPool2D:
# pool_size	= window size over which to take the maximum

class Unet:
    def __init__(self, input_shape):
        self._inputs = Input(input_shape)             

    def conv_block(self, input, num_filters):
        x = Conv2D(num_filters, 3, padding="same")(input)
        # x = BatchNormalization()(x)
        x = InstanceNormalization()(x)
        x = Activation("relu")(x)
        x = Conv2D(num_filters, 3, padding="same")(x)
        # x = BatchNormalization()(x)
        x = InstanceNormalization()(x)
        x = Activation("relu")(x)
        return x

    def encoder_block(self, input, num_filters):
        x = self.conv_block(input, num_filters)
        p = MaxPool2D((2, 2))(x)
        return x, p

    def decoder_block(self, input, skip_features, num_filters):
        x = Conv2DTranspose(num_filters, (2, 2), strides=2, padding="same")(input)
        x = Concatenate()([x, skip_features])
        x = self.conv_block(x, num_filters)
        return x

    def build_net(self):
        # encoder
        s1, p1 = self.encoder_block(self._inputs, 64) # img size = SIZE / 2
        s2, p2 = self.encoder_block(p1, 128) # img size = SIZE / 4
        s3, p3 = self.encoder_block(p2, 256) # img size = SIZE / 8
        s4, p4 = self.encoder_block(p3, 512) # img size = SIZE / 16
        s5, p5 = self.encoder_block(p4, 1024) # img size = SIZE / 32

        # bridge
        b1 = self.conv_block(p5, 2048) # img size = SIZE / 16
        
        # decoder
        d1 = self.decoder_block(b1, s5, 1024) # img size = SIZE / 16
        d2 = self.decoder_block(d1, s4, 512) # img size = SIZE / 8
        d3 = self.decoder_block(d2, s3, 256) # img size = SIZE / 4
        d4 = self.decoder_block(d3, s2, 128) # img size = SIZE / 2
        d5 = self.decoder_block(d4, s1, 64) # img size = SIZE

        # last conv to change the number of output channels
        outputs = Conv2D(1, 1, padding="same", activation="sigmoid")(d5)

        print("Input: ", self._inputs)
        print("Output: ", outputs)

        model = Model(self._inputs, outputs, name="U-Net")
        return model
