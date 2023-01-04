"""
Script to perform all the augmenations mentioned in :
Referred from: https://arxiv.org/pdf/2002.05709.pdf (Appendix A corresponding GitHub: https://github.com/google-research/simclr/)
"""

# Basic Imports
import tensorflow as tf
import tensorflow_addons as tfa
import functools

# Constants to control augmentation strength
FLAGS_color_jitter_strength = 0.5
FLAGS_min_object_covered = 0.3 # min area in the specified boudning box to cover
FLAGS_area_range = (0.3, 1.0) # the cropped area must contain a fraction of the supplied image within this range

# pipe for random rotations
aug_layers = tf.keras.Sequential([tf.keras.layers.RandomRotation(0.1),])

def set_aug_strength_value(cjs, m_obj_cov, a_range): 
    """
    Function to setup the value of the augmentation variables
    Arguments:
        cjs : (float) strength of color jittering to apply to vary the color histograms
        m_obj_cov : (float)  min area in the specified boudning box to cover
        a_range : (tuple) the cropped area must contain a fraction of the supplied image within this range
    Returns :
        None, only sets the values
    """
    global FLAGS_color_jitter_strength
    FLAGS_color_jitter_strength = cjs

    global FLAGS_min_object_covered
    FLAGS_min_object_covered = m_obj_cov

    global FLAGS_area_range
    FLAGS_area_range = a_range

def random_apply(func, p, x):
    """
    Function that acts as a wrapper to apply augmentations randomly
    -> tf cond is a functions whose first argument is a predicate, when it is true then it returns first function (i.e. apply augmentation) else dont apply augmentation
    -> tf less return true if arg1 < arg2 
    -> tf random uniform picks a number between 0 & 1 randomly
    Arguments:
        func : (function) augmentation to apply randomly
        p : (float) probability value [0-1]
        x : (tf.image) input type
    Returns:
        image after applying the specified augmentation stochastically
    """ 
    return tf.cond(tf.less(tf.random.uniform([], minval=0, maxval=1, dtype=tf.float32), tf.cast(p, tf.float32)),
                    lambda: func(x),
                    lambda: x)

def distorted_bounding_box_crop(image, 
                                bbox, 
                                min_object_covered=0.1, 
                                aspect_ratio_range=(0.75, 1.33), 
                                area_range=(0.05, 1.0), 
                                max_attempts=100, 
                                scope=None):
    """
    Generates cropped_image using one of the bboxes randomly distorted.
    See `tf.image.sample_distorted_bounding_box` for more documentation.
    Arguments:
        image: `Tensor` of image data.
        bbox: `Tensor` of bounding boxes arranged `[1, num_boxes, coords]` where each coordinate is [0, 1) and the coordinates are arranged
                as `[ymin, xmin, ymax, xmax]`. If num_boxes is 0 then use the whole image.
        min_object_covered: An optional `float`. Defaults to `0.1`. The cropped area of the image must contain at least this fraction of any bounding box supplied.
        aspect_ratio_range: An optional list of `float`s. The cropped area of the image must have an aspect ratio = width / height within this range.
        area_range: An optional list of `float`s. The cropped area of the image must contain a fraction of the supplied image within in this range.
        max_attempts: An optional `int`. Number of attempts at generating a cropped region of the image of the specified constraints. After `max_attempts` failures, return the entire image.
        scope: Optional `str` for name scope.
    Returns:
        (cropped image `Tensor`, distorted bbox `Tensor`).
        """
    with tf.name_scope('distorted_bounding_box_crop'):
        shape = tf.shape(image)
        sample_distorted_bounding_box = tf.image.sample_distorted_bounding_box(shape,
                                                                                bounding_boxes=bbox,
                                                                                min_object_covered=min_object_covered,
                                                                                aspect_ratio_range=aspect_ratio_range,
                                                                                area_range=area_range,
                                                                                max_attempts=max_attempts,
                                                                                use_image_if_no_bounding_boxes=True)
        bbox_begin, bbox_size, _ = sample_distorted_bounding_box

        # Crop the image to the specified bounding box.
        offset_y, offset_x, _ = tf.unstack(bbox_begin)
        target_height, target_width, _ = tf.unstack(bbox_size)
        image = tf.image.crop_to_bounding_box(image, offset_y, offset_x, target_height, target_width)
        return image

def crop_and_resize(image, height, width):
    """
    Make a random crop and resize it to height `height` and width `width`.
    Arguments:
        image: Tensor representing the image.
        height: Desired image height.
        width: Desired image width.
    Returns:
        A `height` x `width` x channels Tensor holding a random crop of `image`.
    """
    
    bbox = tf.constant([0.0, 0.0, 1.0, 1.0], dtype=tf.float32, shape=[1, 1, 4])
    aspect_ratio = width / height
    # Flag values are the global ones & they are set by set_aug_strength_value
    image = distorted_bounding_box_crop(image,
                                        bbox,
                                        min_object_covered = FLAGS_min_object_covered ,
                                        aspect_ratio_range=(3. / 4 * aspect_ratio, 4. / 3. * aspect_ratio),
                                        area_range= FLAGS_area_range,
                                        max_attempts=100,
                                        scope=None)
    
    return tf.image.resize([image], [height, width], method = "bicubic")[0]

def random_crop_with_resize(image, height, width, p=1.0):
    """
    Randomly crop and resize an image.
    Arguments:
        image: `Tensor` representing an image of arbitrary size.
        height: Height of output image.
        width: Width of output image.
        p: Probability of applying this transformation.
    Returns:
        A preprocessed image `Tensor`.
    """
    def _transform(image):  # pylint: disable=missing-docstring
        image = crop_and_resize(image, height, width)
        return image
    
    return random_apply(_transform, p=p, x=image)

def color_distortion(image, s=1.0):
    
    # image is a tensor with value range in [0, 1].
    # s is the strength of color distortion.
    def color_jitter(x):
        # one can also shuffle the order of following augmentations
        # each time they are applied.
        x = tf.image.random_brightness(x, max_delta=0.8*s)
        x = tf.image.random_contrast(x, lower=1-0.8*s, upper=1+0.8*s)
        x = tf.image.random_saturation(x, lower=1-0.8*s, upper=1+0.8*s)
        x = tf.image.random_hue(x, max_delta=0.2*s)
        x = tf.clip_by_value(x, 0, 1)
        return x

    def color_drop(x):
        image = tf.image.rgb_to_grayscale(x)
        image = tf.tile(image, [1, 1, 3])
        return image
    
    def non_lin_wrap(x):
        
        input_img = tf.image.convert_image_dtype(tf.expand_dims(x, 0), tf.dtypes.float32)
        flow_shape = [1, input_img.shape[1], input_img.shape[2], 2]
        init_flows = tf.random.normal(shape=flow_shape) * 2.0
        dense_img_warp = tfa.image.dense_image_warp(input_img, init_flows)
        dense_img_warp = tf.squeeze(dense_img_warp, 0)
        return dense_img_warp
        
    def random_rotate(x):
        # either apply 90 rotation or +/- 10 skew
        return tf.cond(tf.less(tf.random.uniform([], minval=0, maxval=1, dtype=tf.float32), tf.cast(0.5, tf.float32)), 
                       true_fn = lambda : tf.image.rot90(x, k=1), 
                       false_fn = lambda : aug_layers(x))
    
    def random_gauss_noise(x):
        noise = tf.random.normal(shape = (x.shape), mean = 0.0, stddev = 0.05, dtype = tf.float32) 
        return x + noise 
        
    def gauss_filter(x):
        return tfa.image.gaussian_filter2d(x, (10,10), sigma = 6.0)
    
    def channel_randomisation(x):
        channels = tf.unstack (x, axis=-1)
        c_order = tf.random.shuffle([0,1,2])
        
        return tf.stack([tf.gather(channels, c_order[0]), 
                         tf.gather(channels, c_order[1]), 
                         tf.gather(channels, c_order[2])], 
                         axis=-1)
    
    def random_overlay_color_masks(x):
        r = tf.random.uniform(shape=[], minval=0, maxval=1)
        g = tf.random.uniform(shape=[], minval=0, maxval=1)
        b = tf.random.uniform(shape=[], minval=0, maxval=1)
        r_channel = tf.fill([x.shape[0],x.shape[1]], r) 
        g_channel = tf.fill([x.shape[0],x.shape[1]], g) 
        b_channel = tf.fill([x.shape[0],x.shape[1]], b)
        
        mask_img = tf.stack([r_channel, g_channel, b_channel], axis=-1)
        
        alpha = tf.random.uniform(shape=[], minval=0.7, maxval=1)
        beta = tf.random.uniform(shape=[], minval=0.1, maxval=0.5)
        return alpha*x + beta * mask_img 

    # randomly apply transformation with probability p.
    image = random_apply(color_jitter, p=0.8, x = image)
    image = random_apply(color_drop, p=0.4, x = image)
    image = random_apply(non_lin_wrap, p=0.1, x = image)
    image = random_apply(gauss_filter, p = 0.1, x = image)
    image = random_apply(random_rotate, p = 0.2, x = image)
    image = random_apply(channel_randomisation, p = 0.2, x = image)
    image = random_apply(random_gauss_noise, p = 0.2, x = image)
    image = random_apply(random_overlay_color_masks, p = 0.4, x = image)
    return image

def preprocess_for_train(image, 
                        height, 
                        width, 
                        color_distort = True, 
                        crop = True):
    
    """
    Preprocesses the given image for training.
    Arguments:
        image  : (tf.image) representing an image of arbitrary size.
        height : (int) height of the image to be returned
        width  : (int) width of the image to be returned
        color_distort : (bool) whether to apply the color distortion.
        crop :(bool) Whether to crop the image.
    Returns:
        A preprocessed image `Tensor`.
    """

    # Applying image crop randomly 
    if crop:
        image = random_crop_with_resize(image, height, width)
    

    # Applying color distortion, it becomes necessary when dealing with random crops
    if color_distort:
        image = color_distortion(image, s = FLAGS_color_jitter_strength)
    
    # Safely reshaping the image for batch formation & model inference
    image = tf.reshape(image, [height, width, 3])

    # Clipping steps for image normalisation
    image = tf.clip_by_value(image, 0., 1.)

    return image

def preprocess_image(image, 
                     height, 
                     width, 
                     color_distort = True, 
                     cjs = 0.5, 
                     m_obj_cov = 0.3, 
                     a_range = (0.3,1.0)):
    """
    Preprocesses the given image.
    Arguments:
        image  : (numpy array) representing an image of arbitrary size.
        height : (int) height of the image to be returned
        width  : (int) width of the image to be returned
        color_distort : (bool) whether to apply the color distortion.
        cjs : (float) color jitter strength & the default value is the base line we have
        m_obj_cov : (float)  min area in the specified boudning box to cover
        a_range : (tuple) the cropped area must contain a fraction of the supplied image within this range
    Returns:
        A preprocessed image tf.image of range [0, 1] normalised.
        """

    # updating global values of the augmentation strength control parameters
    set_aug_strength_value(cjs, m_obj_cov, a_range)
    # Np to tensor & data-type setup
    image = tf.image.convert_image_dtype(image, dtype=tf.float32)

    return preprocess_for_train(image, 
                                height, 
                                width, 
                                color_distort = True,
                                crop = True)
 