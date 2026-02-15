# MongoDB Atlas Setup Guide

MongoDB Atlas is MongoDB's cloud database service. Here's how to set it up for your ASD Detection System.

## Step 1: Create MongoDB Atlas Account

1. Go to [MongoDB Atlas](https://www.mongodb.com/cloud/atlas/register)
2. Sign up with Google, GitHub, or email
3. Complete the registration

## Step 2: Create a Cluster

1. Click **"Build a Database"** or **"Create"**
2. Choose **FREE** tier (M0 Sandbox)
   - Provider: AWS, Google Cloud, or Azure (your choice)
   - Region: Choose closest to your location
3. Cluster Name: `asd-detection-cluster` (or any name)
4. Click **"Create Cluster"** (takes 1-3 minutes)

## Step 3: Create Database User

1. Click **"Database Access"** in the left sidebar (under Security)
2. Click **"+ Add New Database User"**
3. Authentication Method: **Password**
4. Username: `asd_admin` (or any username)
5. Password: Click **"Autogenerate Secure Password"** and **SAVE IT!**
6. Database User Privileges: **Read and write to any database**
7. Click **"Add User"**

## Step 4: Configure Network Access

1. Click **"Network Access"** in the left sidebar
2. Click **"+ Add IP Address"**
3. For development:
   - Click **"Allow Access from Anywhere"** (0.0.0.0/0)
   - Confirm
4. For production:
   - Add your specific IP addresses
5. Click **"Confirm"**

## Step 5: Get Connection String

1. Go back to **"Database"** (left sidebar)
2. Click **"Connect"** on your cluster
3. Choose **"Connect your application"**
4. Driver: **Python** (version doesn't matter)
5. Copy the connection string, it looks like:
   ```
   mongodb+srv://asd_admin:<password>@asd-detection-cluster.xxxxx.mongodb.net/?retryWrites=true&w=majority
   ```

## Step 6: Update Environment Variables

Replace `<password>` with your actual password from Step 3.

Update your `.env` file:

```bash
# MongoDB Atlas
MONGODB_URL=mongodb+srv://asd_admin:YOUR_PASSWORD_HERE@asd-detection-cluster.xxxxx.mongodb.net/?retryWrites=true&w=majority&appName=asd-detection-cluster
DATABASE_NAME=asd_detection

# JWT (keep these)
JWT_SECRET_KEY=your-super-secret-key-here
JWT_ALGORITHM=HS256
JWT_EXPIRATION_MINUTES=30
```

**Important**: 
- Replace `YOUR_PASSWORD_HERE` with your actual password
- If your password has special characters, URL-encode them:
  - `@` → `%40`
  - `#` → `%23`
  - `$` → `%24`
  - etc.

## Step 7: Test Connection

Update `src/database.py` (already done, just verify):

```python
MONGODB_URL = os.getenv("MONGODB_URL", "mongodb://localhost:27017")
DATABASE_NAME = os.getenv("DATABASE_NAME", "asd_detection")
```

## Step 8: Run Your Application

```bash
# No need to run local MongoDB anymore!
python run_api.py
```

You should see:
```
✓ Connected to MongoDB at mongodb+srv://...
```

## Step 9: View Your Data (Optional)

1. In Atlas, go to **"Database"** → **"Browse Collections"**
2. You'll see your `asd_detection` database
3. Collections will appear as you create users
4. You can manually view, edit, or delete documents here

## MongoDB Compass (Desktop GUI - Optional)

1. Download [MongoDB Compass](https://www.mongodb.com/try/download/compass)
2. Use the same connection string from Step 5
3. Connect and browse your database visually

## Common Connection String Formats

### Atlas (Cloud)
```
mongodb+srv://username:password@cluster.xxxxx.mongodb.net/database?retryWrites=true&w=majority
```

### Local MongoDB
```
mongodb://localhost:27017
```

### Local with Auth
```
mongodb://username:password@localhost:27017/database?authSource=admin
```

## Pricing

- **M0 (Free)**: 512MB storage, shared resources, perfect for development
- **M10+ (Paid)**: Dedicated clusters, starts at $0.08/hour (~$57/month)
- **Production**: Consider M10 or higher for reliability

## Security Best Practices

✅ **DO:**
- Use strong, unique passwords
- Restrict IP addresses in production
- Enable encryption at rest (available in paid tiers)
- Use role-based access control
- Regularly rotate credentials

❌ **DON'T:**
- Commit connection strings to Git
- Use "Allow Access from Anywhere" in production
- Share database credentials
- Use default passwords

## Backup & Restore

Atlas Free Tier:
- No automatic backups
- Manually export data if needed

Atlas Paid Tiers:
- Automatic continuous backups
- Point-in-time recovery
- Download snapshots

## Migration from Local to Atlas

If you have local data:

```bash
# Export from local
mongodump --uri="mongodb://localhost:27017/asd_detection" --out=./backup

# Import to Atlas
mongorestore --uri="mongodb+srv://user:pass@cluster.mongodb.net/asd_detection" ./backup/asd_detection
```

## Monitoring

In Atlas Dashboard:
- **Metrics**: View database operations, connections, memory usage
- **Performance Advisor**: Get index suggestions
- **Alerts**: Set up email alerts for issues

## Troubleshooting

### "Connection Timeout"
- Check IP whitelist in Network Access
- Verify connection string (username/password)

### "Authentication Failed"
- Double-check username and password
- URL-encode special characters in password

### "Network Error"
- Ensure cluster is running (green status)
- Check firewall/VPN settings

## Next Steps

1. ✅ Set up connection string
2. ✅ Test authentication endpoints
3. ✅ Create a test user
4. Monitor usage in Atlas dashboard
5. Consider upgrading for production use

---

Your app is now using MongoDB Atlas cloud database! 🎉
