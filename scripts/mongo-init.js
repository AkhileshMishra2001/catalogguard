const target = db.getSiblingDB('catalogguard');
target.createUser({user:'catalogguard',pwd:process.env.MONGO_APP_PASSWORD,roles:[{role:'readWrite',db:'catalogguard'}]});
