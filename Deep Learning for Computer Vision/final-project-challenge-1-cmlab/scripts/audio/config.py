
class Config():
    def __init__(self):
        self.batch_size = 32
        self.init_lr = 2e-4
        self.num_epochs = 20
        self.grad_clip = 0.1
        # self.backbone = 'resnet18'
        self.scheduler = 'exp'
        self.data_path = './student_data'  # '/project/g/Ego4D/student_data'
        self.model_path = './audio_model.pt'
        self.output_path = './output.csv'
        self.num_workers = 8
        self.device = 'cuda'
        self.seed = 42
        self.benchmark = 0.65
