from pyts.image import MarkovTransitionField

def getMTF(data, image_size=256, n_bins=10, strategy='uniform'):
    mtf = MarkovTransitionField(image_size=image_size, n_bins=n_bins)
    mtf_image = mtf.fit_transform(data)
    return mtf_image