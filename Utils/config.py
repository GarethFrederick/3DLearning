
class Config(object):
    def __init__(self):
        pass

    @staticmethod
    def _device_suffix(cuda):
        """Map a device string to the numeric suffix used in DDPM/data filenames.

        'cuda:N' → 'N'.  'cpu' (or any string without ':') → '0', so jobs running
        on a CPU-only partition (fitc/fitcq) still locate the existing
        cuda-0-trained DDPM checkpoints and generated-data CSVs.
        """
        return cuda.split(":")[1] if ":" in cuda else "0"

    def diffusion_model_path(self,folder,model_name,cuda,epoch,batch_size,pic_size,T,lr,beta_1,beta_T,**kwargs):
        model_path = f'{folder}/model_{model_name}_cuda-{self._device_suffix(cuda)}_epoch-{epoch}_batch-{batch_size}_picsize-{pic_size}_T-{T}_lr-{lr}_beta1-{beta_1}_betaT-{beta_T}.pth'
        return model_path


    def generate_data_path(self,folder,model_name,cuda,epoch,batch_size,pic_size,T,beta_1,beta_T,**kwargs):

        data_path = rf'{folder}/data_{model_name}_cuda-{self._device_suffix(cuda)}_epoch-{epoch}_batch-{batch_size}_picsize-{pic_size}_T-{T}_beta1-{beta_1}_betaT-{beta_T}.csv'

        return data_path


    def ml_model_path(self,folder,model_name,cuda,epoch,lr,**kwargs):
        # <name>-epoch<N>.pth. cuda/lr stay in the signature for callers but not in
        # the name: checkpoints load on any device via map_location.
        return rf'{folder}/{model_name}-epoch{epoch}.pth'

    def trajectory_data_path(self,folder,model_name,cuda,epoch,batch_size,pic_size,T,beta_1,beta_T,**kwargs):
        data_path = rf'{folder}/data_{model_name}_cuda-{self._device_suffix(cuda)}_epoch-{epoch}_batch-{batch_size}_picsize-{pic_size}_T-{T}_beta1-{beta_1}_betaT-{beta_T}_trajectory.npy'

        return data_path



    def dro_save_model_path(self,folder,dro_name,model_name,cuda,epoch,lr,**kwargs):
        # <dro>-<sub>-epoch<N>.pth, e.g. 3DLearning-DDPM-epoch15.pth.
        return rf'{folder}/{dro_name}-{model_name}-epoch{epoch}.pth'




