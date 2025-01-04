from MOMA_LRG_tools import extract_frames, write_dataset_level, write_object_level, write_relation_level, extract_object_images, extract_object_features, write_adjacent, write_relation_semantic_1

dataset_dir = "/path/to/MOMA_LRG/"

source_all_frame_path = dataset_dir + "interaction/"
dst_extract_frame_path = dataset_dir + "extract_frames/"
fps_file = dataset_dir + 'fps.txt'
dataset_level_file = dataset_dir + 'dataset-level.txt'
dataset_level_train_file = dataset_dir + 'dataset-level-train.txt'
dataset_level_val_file = dataset_dir + 'dataset-level-val.txt'
dataset_level_test_file = dataset_dir + 'dataset-level-test.txt'
object_level_path = dataset_dir + 'video-level-object/'
relation_level_path = dataset_dir + 'video-level-relation-A-O/'
object_image_path = dataset_dir + 'object_images/'
object_features_path = dataset_dir + 'object_features/'
adjacent_path = dataset_dir + 'adjacent/'
semantic_file = dataset_dir + 'relation-semantic_1.txt'

split_train_file = './MOMA_LRG_anns/split_by_trim/train.txt'
split_val_file = './MOMA_LRG_anns/split_by_trim/val.txt'

write_dataset_level(dataset_level_file, '')
write_dataset_level(dataset_level_train_file, 'train')
write_dataset_level(dataset_level_val_file, 'val')
write_dataset_level(dataset_level_test_file, 'test')

extract_frames(source_all_frame_path, dst_extract_frame_path)

write_object_level(object_level_path)

write_relation_level(relation_level_path)

extract_object_images(object_image_path, dataset_level_file, dst_extract_frame_path)

extract_object_features(object_image_path, object_features_path)

write_adjacent(adjacent_path, dataset_level_file, object_level_path, relation_level_path)

write_relation_semantic_1(semantic_file)