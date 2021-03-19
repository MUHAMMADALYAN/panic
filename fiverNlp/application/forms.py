import os

from flask_wtf import FlaskForm
from flask_wtf.file import FileField, FileRequired
from wtforms import SubmitField, TextAreaField, SelectField, StringField, BooleanField
from wtforms.validators import DataRequired, Length

from util_functions import class_arr


class FileInputForm(FlaskForm):
	file   = FileField("Upload CSV file", validators=[FileRequired('File was Empty!')])
	submit = SubmitField("Upload")


class PredictionDataForm(FlaskForm):
	text_area = TextAreaField()
	submit 	  =  SubmitField("Classify")

class TrainModelForm(FlaskForm):
	train = SubmitField("Train Model form Uploaded File")


class ChangeClassColorsForm(FlaskForm):
	new_purpose 	 = StringField("Purpose:", validators=[DataRequired(), Length(min=7, max=7)])
	new_craftsmaship = StringField("Craftsmaship:", validators=[DataRequired(), Length(min=7, max=7)])
	new_aesthetic 	 = StringField("Aesthetic:", validators=[DataRequired(), Length(min=7, max=7)])
	new_narrative 	 = StringField("Narrative:", validators=[DataRequired(), Length(min=7, max=7)])
	submit 		 	 = SubmitField("Set Colors")


class SubmitAllForm(FlaskForm):
	submit = SubmitField("Save")


def special_form(labels):

	class F(FlaskForm): 
		n_attrs = len(labels)
		
		
	for i, label in enumerate(labels):
		setattr(
			F, 
			f"special_{i}", 
			SelectField(
				u"Select Correct Labels", 
				choices   = list(zip(class_arr, class_arr)),
				default   = label,
				validators= [DataRequired()],
			))
	F.proceed   = SubmitField("Proceed")
	
	return F()
