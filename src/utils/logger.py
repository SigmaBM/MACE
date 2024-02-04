import sys


class Logger(object):
    def __init__(self, run_dir, stdout=True):
        self.terminal = sys.stdout
        self.log = open(str(run_dir / 'log.txt'), 'w')
        sys.stdout = self
        
        self.stdout = stdout
        
    def write(self, message):
        if self.stdout:
            self.terminal.write(message)
        self.log.write(message)
        self.log.flush()
        
    def flush(self):
        pass
    
    def close(self):
        self.log.close()
        sys.stdout = self.terminal
    
    def __del__(self):
        self.close()
