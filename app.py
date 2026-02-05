from flask import Flask, request, render_template
from recommender import recommend_jobs

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def home():
    jobs = None
    if request.method == 'POST':
        skills = request.form['skills']
        jobs = recommend_jobs(skills)
    return render_template('index.html', jobs=jobs)

if __name__ == '__main__':
    app.run(debug=True)
