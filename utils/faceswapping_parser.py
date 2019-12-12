import argparse
class faceswapping_parser(argparse.ArgumentParser):
    def __init__(self,description=None):
        super(faceswapping_parser,self).__init__(description=description)
        self._add('--json_file',default="",dest="json_file",help="this is json file to set arguments")

    def _add(self,*arg,**kwargs):
        super(faceswapping_parser,self).add_argument(*arg,**kwargs)
    def parse_args(self,*arg,**kwargs):
        return super(faceswapping_parser,self).parse_args(*arg,**kwargs)
    def value(self):
        return vars(self.parse_args())
if __name__=="__main__":
        arg=faceswapping_parser()
        print(arg)
        print(arg.value())