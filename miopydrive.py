
import argparse
#from pydrive.auth import GoogleAuth
#from pydrive.drive import GoogleDrive


parser = argparse.ArgumentParser()

parser.add_argument('--drive', required=True)

args = parser.parse_args()


drive = pickle.load(open(args.drive,'r'))

file5 = drive.CreateFile()

with open('my.txt','w') as f:
    f.write('ECCOOOO')

file5.SetContentFile('my.txt')
file5.Upload() 

