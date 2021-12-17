from model.HotDogModel import HotDogModel
from utils.utils import load_model_dict, load_model_dict_abs


class HotDogNotHotDogInfer:

    def __init__(self):
        self.model = HotDogModel()

    def load_model(self, dict_name, abs_filepath=None):
        try:
            if abs_filepath is None:
                state_dict = load_model_dict(dict_name)
            else:
                state_dict = load_model_dict_abs(abs_filepath)

            self.model.load_state_dict(state_dict)
            self.model.eval()
        except Exception as e:
            print('Failed to load model!', e)

    def infer(self, query):
        y = self.model(query)
        return y
    
    def set_device(self,device):
        self.model = self.model.to(device)
