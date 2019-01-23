import sqlite3
import time
import os

conn = sqlite3.connect('test.db')
c = conn.cursor()
# print "Opened database successfully"
# c = conn.cursor()
# c.execute('''CREATE TABLE Painting
#        (ID INT PRIMARY KEY     NOT NULL,
#        STATUS INT      NOT NULL,
#        RECVPATH        CHAR(50),
#        LOADPATH        CHAR(50));''')
# print "Table created successfully"
# conn.commit()
# conn.close()

val = 1

# cursor = c.execute("SELECT *  from Painting ")
# for row in cursor:
#        print "ID = ", row[0]
#        print "STATUS = ", row[1]
#        print "RECVPATH = ", row[2]
#        print "LOADPATH = ", row[3], "\n"
# cursor = c.execute("SELECT *  from Painting where STATUS = 0") 
# #os.
# for row in cursor:
#        print ""
#        print "ID = ", row[0]
#        print "STATUS = ", row[1]
#        print "RECVPATH = ", row[2]
# #        print "LOADPATH = ", row[3], "\n"
# for row in cursor:
#        #execute&save
#        UPLOAD_FOLDER ='/root/Chinese-painting-transform-/load/'
#        name = UPLOAD_FOLDER + str(row[0])+"_new."+ "jpg"
#        #row[3]=name
#        tmp="python3 inference.py --model ink2real.pb --input "+row[2]+" --output "+name
#        c.execute("UPDATE Painting set STATUS = 1 where ID = "+str(row[0]))
#        os.system(tmp)
#        print "OK"
#        #change statue
#        tmp = "UPDATE Painting set LOADPATH =" +name+ " where ID = "+str(row[0])
#        print tmp
#        #c.execute("UPDATE Painting set STATUS = 1 where ID = "+str(row[0]))
#        c.execute(tmp)
#        c.execute("UPDATE Painting set STATUS = 2 where ID="+str(row[0]))

# cursor = c.execute("SELECT *  from Painting ") 
# for row in cursor:
#        print "ID = ", row[0]
#        print "STATUS = ", row[1]
#        print "RECVPATH = ", row[2]
#        print "LOADPATH = ", row[3], "\n"
# #os.
# # for row in cursor:
# #        print ""
# #        print "ID = ", row[0]
# #        print "STATUS = ", row[1]
# #        print "RECVPATH = ", row[2]
# #        print "LOADPATH = ", row[3], "\n"
# for row in cursor:
#        #execute&save
#        UPLOAD_FOLDER =' /root/Chinese-painting-transform/load/'
#        name = UPLOAD_FOLDER + str(row[0])+"_new."+ "jpg"
#        #row[3]=name
#        tmp="python3 inference.py --model ink2real.pb --input "+row[2]+" --output "+name
#        c.execute("UPDATE Painting set STATUS = 1 where ID = "+str(row[0]))
#        os.system(tmp)
#        print "OK"
#        #change statue
#        tmp = "UPDATE Painting set LOADPATH = " +name+ " where ID = "+str(row[0])
#        print tmp
#        #c.execute("UPDATE Painting set STATUS = 1 where ID = "+str(row[0]))
#        c.execute(tmp)
#        c.execute("UPDATE Painting set STATUS = 2 where ID="+str(row[0]))

# cursor = c.execute("SELECT *  from Painting ") 
# for row in cursor:
#        print "ID = ", row[0]
#        print "STATUS = ", row[1]
#        print "RECVPATH = ", row[2]
#        print "LOADPATH = ", row[3], "\n"

while val == 1:
       #print "OK"
       cursor = c.execute("SELECT *  from Painting where STATUS = 0")
       #os.
       # for row in cursor:
       #        print ""
       #        print "ID = ", row[0]
       #        print "STATUS = ", row[1]
       #        print "RECVPATH = ", row[2]
       #        print "LOADPATH = ", row[3], "\n"
       for row in cursor:
              #execute&save
              UPLOAD_FOLDER ='/root/Chinese-painting-transform-/load/'
              name = UPLOAD_FOLDER + str(row[0])+"_new."+ "jpg"
              #row[3]=name
              tmp="python3 inference.py --model ink2real.pb --input "+row[2]+" --output "+name
              c.execute("UPDATE Painting set STATUS = 1 where ID = "+str(row[0]))
              os.system(tmp)
              print "OK"
              #change statue
              tmp = "UPDATE Painting set LOADPATH ='" +name+ "' where ID = "+str(row[0])
              print tmp
              #c.execute("UPDATE Painting set STATUS = 1 where ID = "+str(row[0]))
              c.execute(tmp)
              c.execute("UPDATE Painting set STATUS = 2 where ID="+str(row[0]))
              conn.commit()
       time.sleep(5)

