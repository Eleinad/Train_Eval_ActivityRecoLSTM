
import argparse
import pickle
from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive



drive = pickle.load(open('auth.pickle','rb'))

file5 = drive.CreateFile()

with open('my.txt','w') as f:
    f.write('ECCOOOO')

file5.SetContentFile('my.txt')
file5.Upload() 

