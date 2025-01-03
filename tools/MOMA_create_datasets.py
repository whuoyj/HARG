from MOMA_tools import extract_frames, write_dataset_level, write_object_level, write_relation_level, extract_object_images, extract_object_features, write_adjacent

source_video_path = "/path/to/MOMA-1.0/trim_videos/"
dst_frame_path = "/path/to/MOMA-1.0/all_frames/"

fps_file = './fps.txt'
split_train_file = './split_by_trim/train.txt'
split_val_file = './split_by_trim/val.txt'
dataset_level_train_file = './dataset-level-train.txt'
dataset_level_val_file = './dataset-level-val.txt'
object_level_path = './video-level-object/'
relation_level_path = './video-level-relation-A-O/'
object_image_path = './object_images/'
object_features_path = './object_features/'
adjacent_path = './adjacent/'

extract_frames(source_video_path, dst_frame_path, fps_file)

write_dataset_level(dataset_level_train_file, split_train_file, fps_file)
write_dataset_level(dataset_level_val_file, split_val_file, fps_file)

write_object_level(object_level_path, split_train_file, fps_file, dst_frame_path)
write_object_level(object_level_path, split_val_file, fps_file, dst_frame_path)

write_relation_level(relation_level_path, split_train_file, fps_file, dst_frame_path)
write_relation_level(relation_level_path, split_val_file, fps_file, dst_frame_path)

extract_object_images(object_image_path, split_train_file, fps_file, dataset_level_train_file, dst_frame_path)
extract_object_images(object_image_path, split_val_file, fps_file, dataset_level_val_file, dst_frame_path)

extract_object_features(object_image_path, object_features_path)

write_adjacent(adjacent_path, fps_file, dataset_level_train_file, object_level_path, relation_level_path, dst_frame_path)
write_adjacent(adjacent_path, fps_file, dataset_level_val_file, object_level_path, relation_level_path, dst_frame_path)