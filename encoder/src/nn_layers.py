import theano.tensor as T
import lasagne
import theano

class IntLayer(lasagne.layers.Layer):
    def get_output_for(self, input, **kwargs):
        #print_func = theano.printing.Print("d")
        #input = print_func(input)
        return T.cast(input, 'int32')

    def get_output_shape_for(self, input_shape):
        return input_shape

class PrintLayer(lasagne.layers.Layer):
    def get_output_for(self, input, **kwargs):
        print_func = theano.printing.Print("d")
        input = print_func(input)
        return input

    def get_output_shape_for(self, input_shape):
        return input_shape
