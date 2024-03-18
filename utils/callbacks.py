from tqdm.auto import tqdm
from stable_baselines3.common.callbacks import BaseCallback


class ProgressBarCallback(BaseCallback):
    """
    A base callback that updates a progress bar.

    :param pbar: A tqdm.pbar object.
    """

    def __init__(self, pbar):
        """
        Initializes the ProgressBarCallback class.

        :param pbar: A tqdm.pbar object.
        """
        super(ProgressBarCallback, self).__init__()
        self._pbar = pbar

    def _on_step(self):
        """
        Updates the progress bar.

        :return: A boolean that indicates whether to continue training or not.
        """
        # Update the progress bar:
        self._pbar.n = self.num_timesteps
        self._pbar.update(0)
        return True


# this callback uses the 'with' block, allowing for correct initialisation and destruction
class ProgressBarManager(object):
    """
    A context manager that creates and closes a progress bar.

    :param total_timesteps: An integer that represents the total number of timesteps.
    """
    def __init__(self, total_timesteps):  # init object with total timesteps
        """
        Initializes the ProgressBarManager class.

        :param total_timesteps: An integer that represents the total number of timesteps.
        """
        self.pbar = None
        self.total_timesteps = total_timesteps

    def __enter__(self):  # create the progress bar and callback, return the callback
        """
        Creates the progress bar and callback, and returns the callback.

        :return: A ProgressBarCallback object.
        """
        self.pbar = tqdm(total=self.total_timesteps)

        return ProgressBarCallback(self.pbar)

    def __exit__(self, exc_type, exc_val, exc_tb):  # close the callback
        """
        Closes the progress bar.

        :param exc_type: The exception type.
        :param exc_val: The exception value.
        :param exc_tb: The exception traceback.
        """
        self.pbar.n = self.total_timesteps
        self.pbar.update(0)
        self.pbar.close()
