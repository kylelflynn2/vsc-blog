# Kyle Flynn Portfolio

A professional portfolio website showcasing baseball operations and financial analysis projects.

## Setup Instructions

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the Application

```bash
python app.py
```

The app will be available at `http://localhost:5000`

### 3. Initial Setup

On first run, visit `http://localhost:5000/admin/setup` to create your admin password.

### 4. Admin Access

- Login at `http://localhost:5000/admin/login` with your password
- Manage your profile
- Add, edit, and delete projects
- Upload profile photo and project images

## Features

- **Home Page**: Professional hero section with profile information
- **Blog/Projects**: Showcase your work with project cards
- **Connect**: Contact information and call-to-action
- **Admin Dashboard**: Manage profile and projects
- **Image Upload**: Support for PNG, JPG, GIF, and WebP images (max 16MB)
- **Data Persistence**: JSON-based storage for portability

## File Structure

```
.
├── app.py                    # Flask application
├── requirements.txt          # Python dependencies
├── templates/               # HTML templates
│   ├── base.html           # Base template
│   ├── index.html          # Home page
│   ├── blog.html           # Projects page
│   ├── connect.html        # Contact page
│   ├── admin_login.html    # Admin login
│   ├── admin_setup.html    # Initial password setup
│   └── admin_dashboard.html # Admin panel
├── static/
│   ├── css/
│   │   └── style.css       # Main stylesheet
│   └── uploads/            # Uploaded images
└── data/
    ├── profile.json        # Profile data
    ├── posts.json          # Projects/posts data
    └── config.json         # Admin configuration
```

## Notes

- Update the `secret_key` in `app.py` for production use
- Images are automatically optimized and stored in `static/uploads/`
- Data is stored in JSON files in the `data/` folder
- The application supports concurrent uploads with secure filename handling

## Deployment

For production deployment:
1. Change `debug=False` in `app.py`
2. Use a production WSGI server (gunicorn, waitress, etc.)
3. Set a strong `secret_key` in the configuration
4. Consider using a proper database for scalability
