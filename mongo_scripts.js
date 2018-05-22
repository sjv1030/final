db.createCollection('monthlyvalues')

db.monthlyvalues.deleteMany({})

db.monthlyvalues.find().limit(5)

db.monthlyvalues.find({'month_timestamp':{$gt:new Date("2011-01-01")},
'rig':{$exists:true}}).limit(5)

db.monthlyvalues.find({'month_timestamp':{$gt:new Date("2011-01-01")},
'wtc_val':{$exists:true}}).limit(5)

db.weeklyyvalues.find().limit(5)
db.values.find().limit(5)
db.monthlyvalues.find().limit(10)

db.values.find({'wtc_val':{'$exists':true}},{'_id':0,'day_timestamp':1,'wtc_val':1})

db.values.find().limit(10)