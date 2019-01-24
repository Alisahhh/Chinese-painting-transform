from flask import Flask
import sqlite3
from flask import request
from werkzeug.utils import secure_filename
import json
from flask import render_template
#from cv2
import base64
app = Flask(__name__)

PaintingCount = 0

@app.route('/')
@app.route('/api/')
def index():
    return render_template('index.html')

@app.route('/api/task',methods=['GET'])
def TaskList():
    conn = sqlite3.connect('test.db')
    c = conn.cursor()
    cursor = c.execute("SELECT ID, STATUS  from Painting")
    ret = [] 
    for i in cursor:
        tmp = {'ID': i[0], 'WTATUS': i[1]} 
        ret.append(tmp)
    ret = json.dumps(ret, indent=4)
    return ret

@app.route('/api/task/create',methods=['POST','GET'])
def CreateTask():
    #print "OK"
    global PaintingCount
    status = 0
    PaintingID=-1
    if request.method == 'POST':
        # print "OK"
        PaintingCount = PaintingCount+1
        f = request.files['file']
	#out = f.resize(512,512,Image.ANTIALIAS) 
        # print "OK"
        UPLOAD_FOLDER ="/root/Chinese-painting-transform-/recv/"
        print UPLOAD_FOLDER[0:]
        name = UPLOAD_FOLDER + str(PaintingCount)+"."+ secure_filename(f.filename)

        print name[0:]
        f.save(name)
        # print "OK"
        PaintingID=PaintingCount
        conn = sqlite3.connect('test.db')
        c = conn.cursor()
        tmp = "INSERT INTO Painting(ID ,STATUS,RECVPATH,LOADPATH ) VALUES("+str(PaintingID)+",0,'"+name+"','')" 
        print tmp
        c.execute(tmp)
        conn.commit()
        conn.close()
    ret = [status,PaintingID]
    # ret.append(status)
    # ret.append(PaintingID)
    ret = json.dumps(ret,indent=4)
    return ret

@app.route('/api/task/<id>/status',methods=['GET'])
def GetTaskStatus(id):
    status=0
    if request.method == 'GET':
        conn = sqlite3.connect('test.db')
        c = conn.cursor()
        cursor = c.execute("SELECT ID, STATUS  from Painting")
        for row in cursor:
            if int(id) == row[0]:
                status = 1
                ret = [status,row[1]]
                ret =  json.dumps(ret,indent=4)
                return ret
    ret = [status,-1]
    ret = json.dumps(ret,indent=4)
    return ret

def return_img_stream(img_local_path):
    img_stream = ''
    print "OK"
#    print type(img_local_path)
    with open(str(img_local_path), 'r') as img_f:
        img_stream = img_f.read()
        img_stream = base64.b64encode(img_stream)
        print "ok"
    return img_stream

@app.route('/api/task/<id>/download',methods=['GET'])
def DownloadPainting(id):
    status=0
    if request.method == 'GET':
        conn = sqlite3.connect('test.db')
        c = conn.cursor()
        cursor = c.execute("SELECT ID, RECVPATH , LOADPATH  from Painting")
        for row in cursor:
            # print type(row[0])
            # print type(int(id))
            # print id == row[0]
            if int(id) == row[0]:
                print row[1]
                img_stream1 = return_img_stream(row[1])
		img_stream2 = return_img_stream(row[2])
                return render_template('showpainting.html',img_stream1=img_stream1,img_stream2=img_stream2)
    ret=[status]
    ret = json.dumps(ret,indent=4)
    return ret

@app.route('/api/task/<id>',methods=['DELETE'])
def DeletePainting(id):
    status=0
    if request.method == 'DELETE':    
        conn = sqlite3.connect('test.db')
        c = conn.cursor()
        c.execute("DELETE from COMPANY where ID="+id)    
        conn.commit()
        conn.close()
        status=1
    ret=[1]
    ret = json.dumps(ret,indent=4)
    return ret


