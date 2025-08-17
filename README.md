Bangladesh Police Image Analysis System
Project Overview
This software assists Bangladesh police investigations by allowing users to upload images, generate descriptions using AI, perform basic image analysis, and store results in a database. It is designed as a software engineering project, demonstrating web development, AI integration, and database management.
Features

Upload images (JPG/PNG) via a web interface.
Generate image descriptions using Hugging Face's BLIP model.
Perform basic analysis (edge detection) using OpenCV.
Store image metadata, descriptions, analysis results, and case IDs in SQLite.
Display analysis history in a table.

Requirements

Python 3.10+
Libraries: flask, pillow, transformers, torch, opencv-python
Web browser (e.g., Chrome, Firefox)

Setup Instructions

Install Python:

Download and install Python 3.10+ from python.org.
Verify: python --version


Create Project Folder:

Create a folder named police_image_analysis.
Set up the structure:police_image_analysis/
├── app.py
├── templates/
│   └── index.html
├── static/
│   └── styles.css
├── uploads/
└── README.md




Install Libraries:

Open a terminal in the police_image_analysis folder.
Run:pip install flask pillow transformers torch opencv-python




Run the Application:

In the terminal, set Flask environment (if needed):
Windows:set FLASK_APP=app.py
set FLASK_ENV=development


Mac/Linux:export FLASK_APP=app.py
export FLASK_ENV=development




Run the app:python app.py


Open a browser and go to http://192.168.0.100:5002/


Usage:

Upload an image (JPG/PNG) and enter a case ID.
View the generated description and analysis.
See all previous analyses in the table.



Notes

The BLIP model downloads (~1GB) on first run; ensure internet access.
Images are stored in the uploads folder.
The SQLite database (investigations.db) is created automatically.
For production, add user authentication and consider PostgreSQL for scalability.

Troubleshooting

Image Upload Fails: Ensure the uploads folder exists.
Model Errors: Verify internet for initial BLIP download.
Port Conflict: Change port if needed: python app.py --port 5001.

Future Enhancements

Add user authentication (e.g., Flask-Login).
Implement advanced object detection (e.g., YOLOv5, after resolving dependency conflicts).
Deploy on Heroku or AWS for multi-user access.

Author
Created for a software engineering project to assist Bangladesh police investigations.