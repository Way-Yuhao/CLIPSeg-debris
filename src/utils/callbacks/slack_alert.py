from typing import Optional
import os
import io
import traceback
import contextlib
import socket
import requests
from functools import wraps
from datetime import datetime
from dotenv import load_dotenv
from lightning.pytorch.callbacks import RichProgressBar, Callback

slack_alert_msg_printed = False

class SlackAlert(Callback):
    """
    Callback for sending a slack alert.
    """

    def __init__(self, exception_only: bool = False, disabled: bool = False,
                 ignore_keyboard_interrupt: bool = False, at_epoch: Optional[int] = None,
                 at_global_step: Optional[int] = None):
        """
        Args:
            exception_only (bool): Flag to indicate if the alert should only be sent on exceptions.
            disabled (bool): Flag to indicate if the callback is disabled.
            ignore_keyboard_interrupt (bool): Flag to indicate if keyboard interrupts should be ignored.
            at_epoch (int): Send slack alert at the end of the specified epoch.
            at_global_step (int): Send slack alert at the specified global step.
        """
        super().__init__()
        self.exception_only = exception_only
        self.pl_module_device = None
        self.exception_occurred = False
        self.disabled = disabled
        self.ignore_keyboard_interrupt = ignore_keyboard_interrupt
        self.at_epoch = at_epoch
        self.at_global_step = at_global_step
        self.hostname = socket.gethostname()
        self.webhook_url = None
        self.configure_slack_alert()

    def configure_slack_alert(self):
        path_to_restore = os.getcwd()
        os.chdir(os.path.dirname(os.path.realpath(__file__)))
        load_dotenv('.env')
        os.chdir(path_to_restore)
        self.webhook_url = os.getenv('SLACK_WEBHOOK_URL')
        if self.webhook_url is None:
            msg = 'SlackAlert: To send alerts to slack, set SLACK_WEBHOOK_URL in .env file under project root directory.'
            yprint(msg)
            self.disabled = True
        return

    def alert(self, message: str):
        data = {'text': message,
                'username': 'Webhook Alert',
                'icon_emoji': ':robot_face:'}
        response = requests.post(self.webhook_url, json=data)
        if response.status_code != 200:
            raise ValueError('Request to slack returned an error %s, the response is:\n%s'
                             % (response.status_code, response.text))
        return

    def on_train_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        """
        Send slack alert at the end of the specified epoch.
        """

        if self.at_epoch is not None and trainer.current_epoch >= self.at_epoch:
            title = f'Training progress: epoch {self.at_epoch} completed'
            self.at_epoch = None
        elif self.at_global_step is not None and trainer.global_step >= self.at_global_step:
            title = f'Training progress: global step {self.at_global_step} completed'
            self.at_global_step = None
        else:
            return
        wandb_run = trainer.logger.experiment
        wandb_url = wandb_run.url
        title += f' for run `{wandb_run.id}`'
        now = datetime.now()
        formatted_time = now.strftime("%Y-%m-%d %H:%M:%S")
        device = str(trainer.strategy.root_device)
        message = (f'*{title}*\nTime completed: {formatted_time}\nHostname: {self.hostname}\nDevice: {device}\n'
                   f'Wandb URL: {wandb_url}\n')
        self.alert(message)
        return

    def on_exception(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", exception: BaseException) -> None:
        """
        Send slack alert on exceptions.
        """
        if self.disabled:
            # raise exception
            return
        stack_trace = traceback.format_exc()
        device = str(trainer.strategy.root_device)
        now = datetime.now().replace(microsecond=0)
        # Prepare the alert message
        title = f'Exception Occurred'
        message = f'*{title}*```{stack_trace}```\nHost: {self.hostname}\nDevice: {device}\nTime: {now}'
        # Send the alert using your alert function
        if isinstance(exception, KeyboardInterrupt) and self.ignore_keyboard_interrupt:
                print('SlackAlert: Encountered keyboard interrupt. Slack alert not sent.\n'
                      'To enable slack alerts on keyboard interrupts, set `ignore_keyboard_interrupt` to False.')
                raise exception
        self.alert(message)
        self.exception_occurred = True
        # raise exception

    def teardown(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", stage: str) -> None:
        """
        Send slack alert on successful teardown.
        """
        if not self.exception_only and not self.exception_occurred and not self.disabled:
            title = f'{stage.capitalize()} completed'
            now = datetime.now() # current time
            formatted_time = now.strftime("%Y-%m-%d %H:%M:%S")
            device = str(trainer.strategy.root_device) # cuda device
            message = f'*{title}*\n```Time completed: {formatted_time}\nHostname: {self.hostname}\nDevice: {device}```'
            self.alert(message)
        return

def alert(message: str):
    """
    Sends a message to a designated slack channel, which a SLACK_WEBHOOK_URL to be set in .env file.
    If webhook URL is not found, the message is printed to stdout in red.
    :param message:
    :return:
    """
    global slack_alert_msg_printed  # print out error msg only once
    path_to_restore = os.getcwd()
    os.chdir(os.path.dirname(os.path.realpath(__file__)))
    load_dotenv('.env')
    os.chdir(path_to_restore)
    webhook_url = os.getenv('SLACK_WEBHOOK_URL')
    if webhook_url is None:
        if not slack_alert_msg_printed:
            msg = 'To send alerts to slack, set SLACK_WEBHOOK_URL in .env file under project root directory.'
            yprint(msg)  # Assuming yprint is a typo and meant print. Adjust as necessary for your logging method.
            yprint('Message routed to stdout')
            slack_alert_msg_printed = True  # Mark the warning as printed
        rprint(message)
        return
    else:
        data = {'text': message,
                'username': 'Webhook Alert',
                'icon_emoji': ':robot_face:'}
        response = requests.post(webhook_url, json=data)
        if response.status_code != 200:
            raise ValueError(
                'Request to slack returned an error %s, the response is:\n%s'
                % (response.status_code, response.text)
            )
        return


def monitor(func):
    """
    Decorator to monitor the execution of a function. If the function fails, the stack trace is printed.
    No message is sent at completion.
    :param func:
    :return:
    """
    def inner(*args, **kwargs):
        try:
            func(*args, **kwargs)
        except Exception as e:
            stack_trace = traceback.format_exc()
            title = 'Code failed'
            alert(f'*{title}*```{stack_trace}```')
            raise e
    return inner


def monitor_complete(func):
    """
    Decorator to monitor the execution of a function. If the function fails, the stack trace is printed.
    If the function succeeds, a message is sent to a designated slack channel.
    :param func:
    :return:
    """
    def inner(*args, **kwargs):
        try:
            func(*args, **kwargs)
            alert(f'Code executed successfully: {func.__name__}')
        except Exception as e:
            stack_trace = traceback.format_exc()
            title = 'Code failed'
            alert(f'*{title}*```{stack_trace}```')
            raise e
    return inner


def capture_stdout(func):
    """
    Suppresses stdout output, unless it contains an error message, in which case a ValueError is raised.
    :param func:
    :return:
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        # Create a StringIO buffer to capture output
        stdout_capture = io.StringIO()
        with contextlib.redirect_stdout(stdout_capture):
            # Call the original function and capture its return value
            retval = func(*args, **kwargs)
        # Retrieve the captured output
        std_output = stdout_capture.getvalue()
        if ('error' in std_output.lower() or 'warning' in std_output.lower()
                or 'failed' in std_output.lower() or 'exception' in std_output.lower()):
            raise ValueError(f'Found error in suppressed stdout from {func.__name__}\n{std_output}')
        # Return both the function's original return value and the captured output
        return retval
    return wrapper


class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def yprint(msg):
    """
    Print to stdout console in yellow.
    :param msg:
    :return:
    """
    print(f"{bcolors.WARNING}{msg}{bcolors.ENDC}")


def rprint(msg):
    """
    Print to stdout console in red.
    :param msg:
    :return:
    """
    print(f"{bcolors.FAIL}{msg}{bcolors.ENDC}")

if __name__ == '__main__':
    s = SlackAlert()