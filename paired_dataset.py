import tensorflow as tf
from magenta.scripts.convert_dir_to_note_sequences import convert_midi

class PairedDataset(object):
    def __init__(self, midi_list, image_list, config, batch_size=16):
        """
        :param midi_list: List of full paths to midi files.
        :param image_list: List of full paths to image files.
        :param config: Magenta Config object
        """
        self.midi_list = midi_list
        self.image_list = image_list
        self.config = config
        self.bs = batch_size
        
        self.create_tf_dataset()
    
    @staticmethod
    def _remove_pad_fn(padded_seq_1, padded_seq_2, padded_seq_3, length):
        if length.shape.ndims == 0:
            return (padded_seq_1[0:length], padded_seq_2[0:length],
                    padded_seq_3[0:length], length)
        else:
            # Don't remove padding for hierarchical examples.
            return padded_seq_1, padded_seq_2, padded_seq_3, length
    
    @staticmethod
    def decode_img(path):
        img = tf.io.read_file(path)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.convert_image_dtype(img, tf.float32)
        return tf.image.resize(img, [64, 64])
    
    
    def create_tf_dataset(self):
        def gen_midi_data():
            for item in self.midi_list:
                yield convert_midi('', '', str(item)).SerializeToString()
        
        ds1 = tf.data.Dataset.from_generator(gen_midi_data, output_types=tf.string)
        ds1 = ds1.map(self.config.note_sequence_augmenter.tf_augment)
        ds1 = ds1.map(self.config.data_converter.tf_to_tensors, num_parallel_calls=tf.data.experimental.AUTOTUNE)\
                 .flat_map(lambda *t: tf.data.Dataset.from_tensor_slices(t))
        ds1 = ds1.map(PairedDataset._remove_pad_fn)
        
        
        ds2 = tf.data.Dataset.from_generator(lambda: iter(self.image_list), output_types=tf.string)
        ds2 = ds2.map(PairedDataset.decode_img)
        
        self.ds = tf.data.Dataset.zip((ds1, ds2))
        # self.ds = self.ds.padded_batch(self.bs, self.ds.output_shapes, drop_remainder=True).repeat()
        
    def reset_dataset(self):
      self.create_tf_dataset()