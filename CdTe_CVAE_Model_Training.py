import tensorflow as tf
import numpy as np
from skimage import io, transform
import matplotlib.pyplot as plt
from scipy.ndimage import rotate
from skimage.filters import gaussian
import time
from IPython import display
from PIL import Image


@tf.keras.utils.register_keras_serializable(package='Custom', name='CVAE')
class CVAE(tf.keras.Model):
    """Deeper Convolutional Variational Autoencoder."""
 
    def __init__(self, latent_dim):
        reduced_size = int(SIZE / 8)  # Further reduction to allow for deeper layers
        super(CVAE, self).__init__()
        self.latent_dim = latent_dim
        
        # Deeper encoder with more convolutional layers
        self.encoder = tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer(input_shape=(SIZE, SIZE, 1)),
                tf.keras.layers.Conv2D(64, kernel_size=3, strides=(2, 2), activation='relu', padding='same'),
                tf.keras.layers.Conv2D(128, kernel_size=3, strides=(2, 2), activation='relu', padding='same'),
                tf.keras.layers.Conv2D(128, kernel_size=3, strides=(2, 2), activation='relu', padding='same'),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(latent_dim + latent_dim),  # Outputting both mean and log-variance for latent space
            ]
        )

        # Deeper decoder with more Conv2DTranspose layers
        self.decoder = tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer(input_shape=(latent_dim,)),
                tf.keras.layers.Dense(units=reduced_size * reduced_size * 64, activation=tf.nn.relu),
                tf.keras.layers.Reshape(target_shape=(reduced_size, reduced_size, 64)),
                tf.keras.layers.Conv2DTranspose(128, kernel_size=3, strides=2, padding='same', activation='relu'),
                tf.keras.layers.Conv2DTranspose(128, kernel_size=3, strides=2, padding='same', activation='relu'),
                tf.keras.layers.Conv2DTranspose(64, kernel_size=3, strides=2, padding='same', activation='relu'),
                tf.keras.layers.Conv2DTranspose(1, kernel_size=3, strides=1, padding='same'),  # No activation to avoid smoothing
            ]
        )

    @tf.function
    def sample(self, eps=None, apply_sigmoid=True):
        if eps is None:
            eps = tf.random.normal(shape=(100, self.latent_dim))
        return self.decode(eps, apply_sigmoid)
 
    def encode(self, x):
        mean, logvar = tf.split(self.encoder(x), num_or_size_splits=2, axis=1)
        return mean, logvar
 
    def reparameterize(self, mean, logvar):
        eps = tf.random.normal(shape=(latent_dim,))
        return eps * tf.exp(logvar * .5) + mean
 
    def decode(self, z, apply_sigmoid=False):
        logits = self.decoder(z)
        if apply_sigmoid:
            probs = tf.sigmoid(logits)
            return probs
        return logits

    def call(self,image):
        mean, logvar = self.encode(image)
        z = self.reparameterize(mean, logvar)
        predictions = self.sample(z)
        return predictions
    
    @classmethod
    def from_config(cls, config):
        latent_dim = config.get('latent_dim', 20)  # Default value if 'latent_dim' is not in the config
        return cls(latent_dim)


optimizer = tf.keras.optimizers.Adam(1e-4)

def log_normal_pdf(sample, mean, logvar, raxis=1):
    log2pi = tf.math.log(2. * np.pi)
    return tf.reduce_sum(
        -.5 * ((sample - mean) ** 2. * tf.exp(-logvar) + logvar + log2pi),
        axis=raxis)

def compute_loss(model, x):
    mean, logvar = model.encode(x)
    z = model.reparameterize(mean, logvar)
    x_logit = model.decode(z)
    cross_ent = tf.nn.sigmoid_cross_entropy_with_logits(logits=x_logit, labels=x)
    logpx_z = -tf.reduce_sum(cross_ent, axis=[1, 2, 3])
    logpz = log_normal_pdf(z, 0., 0.)
    logqz_x = log_normal_pdf(z, mean, logvar)
    return -tf.reduce_mean(logpx_z + logpz - logqz_x)

@tf.function
def train_step(model, x, optimizer):
    """Executes one training step and returns the loss."""
    with tf.GradientTape() as tape:
        loss = compute_loss(model, x)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

def infrance(model, image):
    mean, logvar = model.encode(image)
    z = model.reparameterize(mean, logvar)
    predictions = model.sample(z)
    return predictions

def generate_images(model, epoch, test_sample):
    mean, logvar = model.encode(test_sample)
    z = model.reparameterize(mean, logvar)
    predictions = model.sample(z)
    fig = plt.figure(figsize=(4, 4))

    for i in range(predictions.shape[0]):
        plt.subplot(4, 4, i + 1)
        plt.imshow(predictions[i, :, :, 0], cmap='gray')
        plt.axis('off')
    plt.show()

    fig = plt.figure(figsize=(4, 4))

    for i in range(test_sample.shape[0]):
        plt.subplot(4, 4, i + 1)
        plt.imshow(test_sample[i, :, :, 0], cmap='gray')
        plt.axis('off')
    plt.show()

def predict(model, inp_image, apply_sigmoid=True):
    mean, logvar = model.encode(inp_image)
    z = model.reparameterize(mean, logvar)
    predictions = model.sample(z, apply_sigmoid)
    return predictions[0, :, :, 0]

def get_testing_training_sets(image='20111206DF.tif', size=180, n_images=20):
    """Reads an image file 'image' and crops it in 2*n_images smaller sections of size 'size'."""
    training = []
    testing = []
    img = io.imread(image)
    width = len(img[0])
    height = len(img)
    try:
        img = img[:, :, 0]
    except:
        pass
    for i in range(n_images):
        shift_y = np.random.randint(height / 2, height - size)
        shift_x = np.random.randint(0, width - size)
        training.append(img[shift_y:shift_y + size, shift_x:shift_x + size])
        shift_y = np.random.randint(0, height / 2 - size)
        shift_x = np.random.randint(0, width - size)
        testing.append(img[shift_y:shift_y + size, shift_x:shift_x + size])
    return np.array(training), np.array(testing)

def gaussian_blur(img, sigma):
    """Returns the Gaussian blurred version of the image 'img' with a sigma value of 'sigma'"""
    return np.array(gaussian(img, (sigma, sigma)))

def gaussian_blur_arr(images, sigma):
    """Applies the function gaussian_blur to all images in the set 'images'"""
    a = []
    for img in images:
        a.append(gaussian_blur(img, sigma))
    return np.array(a)

def norm_max_pixel(images):
    """Normalizes each image in the array 'images' such that the pixel intensities are within a range of 0 to 1"""
    a = []
    for img in images:
        img = img / np.max(img)
        a.append(img)
    return np.array(a)

def preprocess_images(images, size, sigma):
    images = gaussian_blur_arr(images, sigma)
    images = norm_max_pixel(images)
    images = images.reshape((images.shape[0], size, size, 1))
    return images.astype('float32')


SIZE = 64 #set the size in pixels of the training samples (previously size=64)
SIGMA = 1 #Set the Gaussian Blurring sigma value
epochs = 3000 #Number of epochs on which to train the CVAE
latent_dim = 20 #Latent dimentions of the CVAE 

im=Image.open("CdTe_Training_Image.tif").convert("L")
print(np.array(im).shape)
im.save('CdTe_Training_Image_L.tif')

image_file = "CdTe_Training_Image_L.tif" #Image file used to obtain the training sets
# =============================================================================
path = r"C:\Users\victa\OneDrive\Documents\GitHub\2025_Hackathon" # Insert the path of training image "CdTe_Training_Image", this will be your working directory
training_set, testing_set = get_testing_training_sets(image_file, SIZE, 100)

train_images = preprocess_images(training_set, SIZE, SIGMA)
test_images = preprocess_images(testing_set, SIZE, SIGMA)

train_size = 16
batch_size = 16
test_size = 5

train_dataset = (tf.data.Dataset.from_tensor_slices(train_images)
                 .shuffle(train_size).batch(batch_size))
test_dataset = (tf.data.Dataset.from_tensor_slices(test_images)
                .shuffle(test_size).batch(batch_size))

num_examples_to_generate = 16
random_vector_for_generation = tf.random.normal(
    shape=[num_examples_to_generate
, latent_dim])

# Call the CVAE and set the desired size of the samples
model = CVAE(latent_dim)
model.build(input_shape=(1, SIZE, SIZE, 1))

# Pick a sample of the test set for generating output images
assert batch_size >= num_examples_to_generate
for test_batch in test_dataset.take(1):
    test_sample = test_batch[0:num_examples_to_generate, :, :, :]

# generate_images(model, 0, test_sample)  # rerun this cell after 50 epochs steps

for epoch in range(1, epochs + 1):
    start_time = time.time()
    for train_x in train_dataset:
        train_step(model, train_x, optimizer)
    end_time = time.time()

    loss = tf.keras.metrics.Mean()
    for test_x in test_dataset:
        loss(compute_loss(model, test_x))
    elbo = -loss.result()
    display.clear_output(wait=False)
    print('Epoch: {}, Test set ELBO: {}, time elapse for current epoch: {}'
          .format(epoch, elbo, end_time - start_time))
    # generate_images(model, epoch, test_sample)

#model.save(path + image_file.strip(".tif") + '_Size{}_SIGMA{}_epochs{}_latentdim{}.keras'.format(SIZE, SIGMA, epochs,
#                                                                                               latent_dim))
model.save(path + image_file.strip(".tif") + '_Size{}_Sigma{}_epochs{}_latentdim{}.keras'.format(SIZE, SIGMA, epochs, latent_dim))

