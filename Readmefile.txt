Install Python 3.6 or higher:
Download the installer from python.org
Run the installer and check "Add Python to PATH"
Verify installation by running "python --version" in a terminal
Install Visual Studio Code:
Download from code.visualstudio.com
Run the installer and follow the prompts
Set up a virtual environment:
Open a terminal in your project directory
Run: python -m venv env
Activate the virtual environment:
Windows: env\Scripts\activate
Mac/Linux: source env/bin/activate
Clone the project repository:
Use git clone or download and extract the ZIP
Open the project in VS Code:
Launch VS Code
Go to File > Open Folder and select the project folder
Select the Python interpreter:
In VS Code, press Ctrl+Shift+P (or Cmd+Shift+P on Mac)
Type "Python: Select Interpreter"
Choose the Python interpreter from your virtual environment (env)
Install dependencies:
In the terminal, run: pip install -r requirements.txt
Run the Flask app:
In the terminal, run: python app.py
Access the app:
Open a web browser
Go to http://localhost:5000
Use the app:
Enter patient data manually or
Upload the provided CSV file for predictions