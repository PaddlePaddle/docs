import urllib2
import os

def getHtml(url):
    html = urllib2.urlopen(url).read()
    return html
 
def saveHtml(file_name, file_content, dire):
 
    with open(dire+file_name, "wb") as f:
  
        f.write(file_content)
 
webpages=['io.html', #0
        'fluid.html', #1
        'profiler.html', #2
        'backward.html', #3
        'data_feeder.html', #4
        'metrics.html', #5
        'clip.html', #6
        'regularizer.html',#7 
        'nets.html', #8
        'executor.html',#9 
        'initializer.html',#10 
        'transpiler.html', #11
        'average.html', #12
        'layers.html', #13
        'optimizer.html']#14

version=raw_input("API HTML version(for example: 1.3 1.4 develop):")


dire=raw_input("download to (directory):")


for filename in webpages:
    aurl = "http://paddlepaddle.org/documentation/docs/en/"+version+"/api/"+filename
    html = getHtml(aurl)
    if(not os.path.isdir(dire)):
        os.mkdir(dire)
    classname=filename[:filename.find(".html")]
    filename=classname+"_en.html"
    saveHtml(filename, html, dire+"/")
    print(version+" downloaded --- "+ filename)

