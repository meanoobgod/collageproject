import os
import uuid
from datetime import datetime
from flask import Flask, render_template, request, redirect, url_for
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import func
from sqlalchemy import text

from app.GetScore import getScore

app = Flask(__name__)

# --- MySQL Database Configuration ---
# Update credentials or pass them as environment variables
DB_USER = os.environ.get("MYSQL_USER", "root")
DB_PASS = os.environ.get("MYSQL_PASSWORD", "Rohit_2006")
DB_HOST = os.environ.get("MYSQL_HOST", "localhost")
DB_PORT = os.environ.get("MYSQL_PORT", "3306")
DB_NAME = os.environ.get("MYSQL_DB", "collageproject")

# Uses pymysql as the database driver
app.config["SQLALCHEMY_DATABASE_URI"] = (
    f"mysql+pymysql://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
)
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

db = SQLAlchemy(app)

# --- Database Models ---

class Paper(db.Model):
    __tablename__ = "papers"
    paper_id = db.Column(db.String(6), primary_key=True)
    title = db.Column(db.String(255), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)

    questions = db.relationship("Question", backref="paper", cascade="all, delete-orphan")
    submissions = db.relationship("Submission", backref="paper", cascade="all, delete-orphan")


class Question(db.Model):
    __tablename__ = "questions"
    id = db.Column(db.String(10), primary_key=True, nullable=False)
    paper_id = db.Column(db.String(6), db.ForeignKey("papers.paper_id"), nullable=False)
    question_no = db.Column(db.Integer, nullable=False)
    question_text = db.Column(db.Text, nullable=False)
    correct_answer = db.Column(db.Text, nullable=False)

    student_answers = db.relationship("StudentAnswer", backref="question", cascade="all, delete-orphan")


class Submission(db.Model):
    __tablename__ = "submissions"
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    paper_id = db.Column(db.String(6), db.ForeignKey("papers.paper_id"), nullable=False)
    student_name = db.Column(db.String(255), nullable=False)
    submitted_at = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)

    answers = db.relationship("StudentAnswer", backref="submission", cascade="all, delete-orphan")


class StudentAnswer(db.Model):
    __tablename__ = "student_answers"
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    submission_id = db.Column(db.Integer, db.ForeignKey("submissions.id"), nullable=False)
    question_id = db.Column(db.String(10), db.ForeignKey("questions.id"), nullable=False)
    submitted_answer = db.Column(db.Text, nullable=True)
    score = db.Column(db.Float, nullable=False)

class Result(db.Model):
    __tablename__ = "result"
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    rollno = db.Column(db.String(30), nullable=False)
    score = db.Column(db.Float, nullable=False)
    paper_id = db.Column(db.String(6))

# Initialize tables in MySQL database
with app.app_context():
    db.create_all()

# --- Routes ---

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/teacherspace")
def teacherspace():
    return render_template("teacherspace.html")

@app.route("/studentspace")
def studentspace():
    return render_template("studentspace.html")

@app.route("/teacherspace/getresult", methods=["GET"])
def GetResultTeacher():
    # Expects paper_id (the 6-character code) from the URL arguments
    arguments = dict(request.args)
    student_results = []
    if arguments.get("test_id"):
        paper_id = arguments.get("test_id")
        student_results = db.session.execute(text("Select * from result where paper_id= :paper_id"), {"paper_id": paper_id}).mappings().all()

    return render_template(
        "getresultteacher.html", 
        student_results=student_results
        )


@app.route("/student/getresult", methods=["GET"])
def GetResultStudent():
    # Expects paper_id (the 6-character code) from the URL arguments
    arguments = dict(request.args)
    student_results = []
    if arguments.get("test_id") and arguments.get("rollno"):
        paper_id = arguments.get("test_id")
        rollno = arguments.get("rollno").lower()
        student_results = db.session.execute(text("Select * from result where paper_id= :paper_id and rollno = :rollno"), {"paper_id": paper_id, "rollno": rollno}).mappings().all()

    return render_template(
        "getresultstudent.html", 
        student_results=student_results
        )

@app.route("/new", methods=["GET", "POST"])
def newQuestionPaper():
    if request.method == "POST":
        paper_id = str(uuid.uuid4())[:6].upper()
        title = request.form.get("title")

        new_paper = Paper(paper_id=paper_id, title=title)
        db.session.add(new_paper)

        for key in request.form:
            if key.startswith("questions[") and key.endswith("][question]"):
                index = key.split("[")[1].split("]")[0]
                q_text = request.form.get(f"questions[{index}][question]")
                ans_text = request.form.get(f"questions[{index}][answer]")
                _id = f"{paper_id}:{index}"

                question_entry = Question(
                    id = _id,
                    paper_id=paper_id,
                    question_no=int(index),
                    question_text=q_text,
                    correct_answer=ans_text,
                )
                db.session.add(question_entry)

        db.session.commit()

        return (
            f"<h3>Paper Created Successfully!</h3>"
            f"<p>Give this Paper ID to your students: "
            f"<b><a href='/student?paper_id={paper_id}'>{paper_id}</a></b></p><br>"
            f"<a href='/'>Go Home</a>"
        )

    return render_template("new_paper.html")


@app.route("/student", methods=["GET", "POST"])
def student_portal():
    student_name = request.args.get("student_name")
    paper_id = request.args.get("paper_id")

    if request.method == "POST":
        paper_id = request.form.get("paper_id").strip().upper()
        student_name = request.form.get("student_name").strip().lower()

        paper_exists = db.session.query(
            db.exists().where(Paper.paper_id == paper_id)
        ).scalar()

        if paper_exists:
            return redirect(
                url_for("exam", paper_id=paper_id, student_name=student_name)
            )
        else:
            return "<h3>Error: Invalid Paper ID</h3><a href='/student'>Try Again</a>"

    return render_template(
        "student_portal.html", student_name=student_name, paper_id=paper_id
    )


@app.route("/exam/<paper_id>", methods=["GET", "POST"])
def exam(paper_id):
    totalScore = 0
    student_name = request.args.get("student_name")
    paper = Paper.query.get_or_404(paper_id)

    questions_list = (
        Question.query.filter_by(paper_id=paper_id)
        .order_by(Question.question_no)
        .all()
    )

    if request.method == "POST":
        submission = Submission(paper_id=paper_id, student_name=student_name)
        db.session.add(submission)
        db.session.flush()

        correct_answers = {q.id: q.correct_answer for q in questions_list}
        #print(request.form.items)
        for q_id, correct_ans in correct_answers.items():
            #print(q_id)
            user_ans = request.form.get(str(q_id), "")
            calculated_score = getScore(user_ans, correct_ans)
            totalScore += calculated_score
            #print(calculated_score)

            answer_entry = StudentAnswer(
                submission_id=submission.id,
                question_id=q_id,
                submitted_answer=user_ans,
                score=calculated_score,
            )
            db.session.add(answer_entry)

        result_entry = Result(rollno=student_name, score=totalScore/len(correct_answers.keys()), paper_id=paper_id)
        db.session.add(result_entry)
        db.session.commit()
        return (
            f"<h3>Test Submitted!</h3>"
            f"<p>Thank you, {student_name}. Your answers have been saved to MySQL.</p>"
            f"<a href='/'>Go Home</a>"
        )

    formatted_questions = [
        {
            "question_id": q.id,
            "question": q.question_text,
            "question_no": q.question_no,
        }
        for q in questions_list
    ]

    return render_template(
        "exam.html",
        Question=formatted_questions,
        student_name=student_name,
        paper_title=paper.title,
        paper_id=paper_id,
    )


if __name__ == "__main__":
    app.run(debug=True)
