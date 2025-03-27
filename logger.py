import logging
import functools
import inspect

#create a logger
logger = logging.getLogger(__name__)

def  _log_config(func):
    
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        #set up logging configuration
        logging.basicConfig(
            level=logging.INFO,
            filename='test.log',
            encoding='UTF-8',
            filemode='a+',
            format='%(asctime)s %(levelname)s %(message)s'
        )
        return func(*args, **kwargs)
    
    return wrapper

@_log_config
def print_debug(message: str):
    '''log a debud message'''
    frame = inspect.stack()[2]
    cls = frame.frame.f_locals.get('self', None)
    if cls:
        cls_name = cls.__class__.__name__
        method_name  = frame.function
        logger.debug(f'{cls_name}.{method_name} - {message}')
    else:
        logger.debug(f'{frame.function} ({frame.filename}:{frame.lineno}) - {message}')