
class Config():
    def __init__(self):
        self.batch_size = 32
        self.init_lr = 1e-4
        self.start_epoch = 0
        self.num_epochs = 100
        self.image_size = 128
        self.grad_clip = 0.1
        self.samples = 5
        self.tta_num = 3
        self.backbone = 'ttm'
        self.scheduler = 'exp'
        self.frame_dir = './student_data/Frame_person2'  # '/project/g/Ego4D/student_data/Frame'
        self.audio_prep_dir = './student_data/Audio_Preprocess'
        self.model_path = './ttm_model213.pt'
        self.ckpt_dir = './checkpoint'

        self.video_dir = ''
        self.seg_dir = ''
        self.bbox_dir = ''
        self.output_csv = ''

        # self.data_path = './student_data'  # '/project/g/Ego4D/student_data'
        # self.output_path = './output.csv'

        self.vis_model = './swint_acc0.6558.pt'
        self.aud_model = './audio_0.6538.pt'
        self.num_workers = 8
        self.device = 'cuda'
        self.seed = 42
        self.benchmark = 0.7
