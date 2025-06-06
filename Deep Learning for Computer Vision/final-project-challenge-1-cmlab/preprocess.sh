# python3 scripts/preprocess/dataPreprocessing_v2.py --video_dir './student_data/videos' --seg_dir './student_data/train/seg' --bbox_dir './student_data/train/bbox' --new_frames_dir './student_data/Frame_person2' --new_audio_dir './student_data/Audio'
# python3 scripts/preprocess/dataPreprocessing_v2.py --video_dir './student_data/videos' --seg_dir './student_data/test/seg' --bbox_dir './student_data/test/bbox' --new_frames_dir './student_data/Frame_person2' --new_audio_dir './student_data/Audio'
# python3 scripts/preprocess/audio_preprocessV2.py --video_dir './student_data/videos' --seg_dir './student_data/test/seg' --new_audio_dir_before './student_data/Audio' --new_audio_dir_after './student_data/Audio_preprocess2'
# python3 scripts/preprocess/audio_preprocessV2.py --video_dir './student_data/videos' --seg_dir './student_data/train/seg' --new_audio_dir_before './student_data/Audio' --new_audio_dir_after './student_data/Audio_preprocess2'

python3 scripts/preprocess/dataPreprocessing_v2.py --video_dir $1 --seg_dir $2 --bbox_dir $3 --new_frames_dir './student_data/Frame_person2' --new_audio_dir './student_data/Audio'
python3 scripts/preprocess/audio_preprocessV2.py --video_dir $1 --seg_dir $2 --new_audio_dir_before './student_data/Audio' --new_audio_dir_after './student_data/Audio_Preprocess'

# bash preprocess.sh './student_data/videos' './student_data/train/seg' './student_data/train/bbox' && bash preprocess.sh './student_data/videos' './student_data/test/seg' './student_data/test/bbox'
