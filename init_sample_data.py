"""
Optional: Initialize sample data for testing the portfolio
Run this once if you want to populate the app with sample data
"""

import json
import os
from werkzeug.security import generate_password_hash

DATA_FOLDER = 'data'
os.makedirs(DATA_FOLDER, exist_ok=True)

# Sample profile data
profile_data = {
    "name": "Kyle Flynn",
    "subtitle": "Baseball Operations & Finance",
    "bio": "Welcome to my portfolio! I'm passionate about baseball operations, financial analysis, and data-driven decision-making. With expertise in sports analytics and organizational finance, I focus on how strategic insights can drive performance improvements.",
    "photo": None
}

# Sample posts data
posts_data = [
    {
        "id": "1",
        "title": "Baseball Analytics Dashboard",
        "description": "Built a comprehensive analytics dashboard using Python and Tableau to track player performance metrics, injury probability, and team dynamics across multiple seasons.",
        "date": "March 15, 2024",
        "project_url": "https://example.com/analytics",
        "github_url": "https://github.com/example/baseball-analytics",
        "tags": ["Python", "Analytics", "Tableau", "Baseball"],
        "image": None
    },
    {
        "id": "2",
        "title": "Payroll Optimization Model",
        "description": "Developed a financial modeling system to optimize team payroll while maintaining competitive performance. Analyzed salary structures, contract valuations, and ROI metrics.",
        "date": "February 20, 2024",
        "project_url": "https://example.com/payroll",
        "github_url": "https://github.com/example/payroll-optimizer",
        "tags": ["Finance", "Modeling", "Economics", "Baseball"],
        "image": None
    }
]

# Config with default password (change this!)
config_data = {
    "admin_password_hash": generate_password_hash("admin123")  # Default password: admin123
}

# Save files
with open(os.path.join(DATA_FOLDER, 'profile.json'), 'w') as f:
    json.dump(profile_data, f, indent=2)

with open(os.path.join(DATA_FOLDER, 'posts.json'), 'w') as f:
    json.dump(posts_data, f, indent=2)

with open(os.path.join(DATA_FOLDER, 'config.json'), 'w') as f:
    json.dump(config_data, f, indent=2)

print("✅ Sample data initialized!")
print("📝 Default admin password: admin123")
print("⚠️  Change this in /admin/settings after first login")
