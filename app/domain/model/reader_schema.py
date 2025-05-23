from dataclasses import dataclass
from flask import json

import pandas as pd
@dataclass
class ReaderSchema:
    def __init__(self):
        
        self._context = 'C://Users//bitcamp//Documents//kpmg-250424//kpmg2501//V2//ai-server//app//stored_data//crime'
        self._fname = ''
    @property
    def context(self) -> str:
        return self._context
    @context.setter
    def context(self,context):
        self._context = context
    @property
    def fname(self) -> str:
        return self._fname
    @fname.setter
    def fname(self,fname):
        self._fname = fname
    def new_file(self)->str:
        return self._context + self._fname
    def csv_to_dframe(self) -> object:
        file = self.new_file()
        return pd.read_csv(file, thousands=',')
    def xls_to_dframe(self, header, usecols)-> object:
        file = self.new_file()
        return pd.read_excel(file, header=header, usecols=usecols)
    def json_load(self):
        file = self.new_file()
        return json.load(open(file))