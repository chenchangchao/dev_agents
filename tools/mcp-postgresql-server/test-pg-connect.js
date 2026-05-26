// src/test-pg-connect.js
import dotenv from 'dotenv'; // Load environment variables from .env file
dotenv.config();
import {Client} from 'pg'; // Import the pg client

async function testDatabaseConnection() {
  const client = new Client({
    connectionString: process.env.DATABASE_URL, // Use DATABASE_URL from environment variables
  });

  try {
    await client.connect(); // Attempt to connect to the database
    console.log('✅ Successfully connected to the PostgreSQL database!');
    
    // Optionally, run a simple query to verify the connection
    const result = await client.query('SELECT NOW()');
    console.log('🕒 Current time from DB:', result.rows[0].now);
  } catch (err) {
    console.error('❌ Failed to connect to the PostgreSQL database:', err.message);
  } finally {
    await client.end(); // Close the connection
  }
}

// Run the test
testDatabaseConnection();