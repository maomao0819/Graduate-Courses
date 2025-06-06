
class Config():
    def __init__(self):
        self.batch_size = 32
        self.init_lr = 2e-4
        self.data_path = './student_data'  # '/project/g/Ego4D/student_data'
        self.num_workers = 8
        self.device = 'cuda'
        self.seed = 42
