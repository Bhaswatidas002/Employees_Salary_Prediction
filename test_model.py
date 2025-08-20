import joblib
import numpy as np


def predict_once(model, enc_gender, enc_dept, gender, years_exp, dept, job_rate):
    gender_enc = enc_gender.transform([gender])[0]
    dept_enc = enc_dept.transform([dept])[0]
    X = np.array([gender_enc, years_exp, dept_enc, job_rate]).reshape(1, -1)
    return float(model.predict(X)[0])


def main():
    print("Loading model and encoders...")
    model = joblib.load('salary_predictor_rf_model.pkl')
    enc_gender = joblib.load('encoder_gender.pkl')
    enc_dept = joblib.load('encoder_department.pkl')

    print("Available departments:", list(enc_dept.classes_))
    print("Available genders:", list(enc_gender.classes_))

    tests = [
        ("Male", 2.0, "IT", 6.0),
        ("Female", 8.0, "Operations", 8.5),
        ("Female", 5.0, "HR", 7.0),
        ("Male", 10.0, "Sales", 7.5),
    ]

    print("\nRunning predictions:")
    for i, (g, y, d, r) in enumerate(tests, 1):
        try:
            pred = predict_once(model, enc_gender, enc_dept, g, y, d, r)
            print(f"{i}. {g}, {y} yrs, {d}, rate {r} -> ${pred:,.2f}/mo  (${pred*12:,.2f}/yr)")
        except Exception as e:
            print(f"{i}. ERROR for ({g}, {y}, {d}, {r}): {e}")


if __name__ == "__main__":
    main()
