import os
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'
import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout
from tensorflow.keras.models import Sequential
import tensorflow_probability as tfp

tfd = tfp.distributions
tfpl = tfp.layers

def create_model(NUM_CLASSES):
    divergence_fn = lambda q,p,_:tfd.kl_divergence(q,p)/10000
    
    model= Sequential()
    model.add(tfpl.Convolution2DReparameterization(input_shape=(28,28,3), filters=32, kernel_size=3, activation='relu',
                                           padding='same',
                                           kernel_prior_fn = tfpl.default_multivariate_normal_fn,
                                           kernel_posterior_fn=tfpl.default_mean_field_normal_fn(is_singular=False),
                                           kernel_divergence_fn = divergence_fn,
                                           bias_prior_fn = tfpl.default_multivariate_normal_fn,
                                           bias_posterior_fn=tfpl.default_mean_field_normal_fn(is_singular=False),
                                           bias_divergence_fn = divergence_fn))
    model.add(Conv2D(32, (3,3), activation='relu'))
    
    model.add(MaxPooling2D(2,2))
    model.add(Dropout(0.25))
    
    model.add(Conv2D(64, (3,3), activation='relu'))
    model.add(MaxPooling2D(2,2))
    
    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.25))
    model.add(tfpl.DenseReparameterization(units=tfpl.OneHotCategorical.params_size(NUM_CLASSES), activation=None,
                    kernel_prior_fn = tfpl.default_multivariate_normal_fn,
                    kernel_posterior_fn=tfpl.default_mean_field_normal_fn(is_singular=False),
                    kernel_divergence_fn = divergence_fn,
                    bias_prior_fn = tfpl.default_multivariate_normal_fn,
                    bias_posterior_fn=tfpl.default_mean_field_normal_fn(is_singular=False),
                    bias_divergence_fn = divergence_fn))
    
    model.add(tfpl.OneHotCategorical(NUM_CLASSES))
    return model
