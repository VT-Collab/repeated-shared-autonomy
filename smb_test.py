import os
from subprocess import Popen, PIPE, STDOUT

# Note: Try with shell=True should not be used, but it increases readability to new users from my years of teaching people Python.
process = Popen('mkdir ~/mnt && mount -t cifs //172.16.0.1/panda-ws ~/mnt -o username=VT-Collab', shell=True, stdout=PIPE, stderr=PIPE)
while process.poll() is None:
    print(process.stdout.readline()) # For debugging purposes, in case it asks for password or anything.

print(os.listdir('~/mnt'))