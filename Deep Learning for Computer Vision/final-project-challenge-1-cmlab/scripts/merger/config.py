
class Config():
    def __init__(self):
        self.batch_size = 64
        self.init_lr = 2e-3
        self.start_epoch = 0
        self.num_epochs = 30
        self.image_size = 128
        self.grad_clip = -1
        self.samples = 5
        self.tta_num = 3
        self.backbone = 'resnet50'  # 'swin_t'
        self.scheduler = 'cos'  # 'exp' or 'cos'
        self.data_path = './student_data'  # '/project/g/Ego4D/student_data'
        self.frame_dir = './student_data/Frame_person2'  # '/project/g/Ego4D/student_data/Frame'
        self.vis_model = './vision_model.pt'
        self.aud_model = './audio_model_best.pt'
        self.fus_model = './fusion_model.pt'
        self.output_path = './output.csv'
        self.num_workers = 8
        self.device = 'cuda'
        self.seed = 42
        self.benchmark = 0.68
