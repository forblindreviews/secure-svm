import subprocess

result = subprocess.run(['ls'], stdout=subprocess.PIPE, shell=True)
print(result.stdout.decode('utf-8'))