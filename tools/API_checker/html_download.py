import urllib2
import os

def getHtml(url):
    html = urllib2.urlopen(url).read()
    return html
 
def saveHtml(file_name, file_content, dire):
 
    with open(dire+file_name, "wb") as f:
  
        f.write(file_content)
 
files=['io_en.html', #0
        'fluid_en.html', #1
        'profiler_en.html', #2
        'backward_en.html', #3
        'data_feeder_en.html', #4
        'metrics_en.html', #5
        'clip_en.html', #6
        'regularizer_en.html',#7 
        'nets_en.html', #8
        'executor_en.html',#9 
        'initializer_en.html',#10 
        'transpiler_en.html', #11
        'average_en.html', #12
        'layers_en.html', #13
        'optimizer_en.html']#14

version=input("API HTML version(for example: 1.3 1.4 develop):")
version=str(version).replace(" ","")

dire=input("download to (directory):")
dire=str(dire).replace(" ","")

for filename in files:
    aurl = "http://paddlepaddle.org/documentation/docs/en/"+version+"/api/"+filename
    html = getHtml(aurl)
    if(not os.path.isdir(dire)):
        os.mkdir(dire)
    saveHtml(filename, html, version+"/")
    print(version+" downloaded --- "+ filename)

'''
for filename in files:
    aurl = "http://paddlepaddle.org/documentation/docs/en/develop/api/"+filename
    html = getHtml(aurl)
    saveHtml(filename, html, "develop/")
    print("develop ok --- "+ filename)
'''