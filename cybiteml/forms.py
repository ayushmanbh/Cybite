from django import forms

YES_NO = [('yes', 'Yes'),
          ('no', 'No')]
HEAD_OR_NOT = [('head', 'Head'), ('full', 'Full frame')]
ALGOS = [('gnb', 'GaussianNB'), ('lrc', 'LogisticRegression'), ('svc', 'Support Vector Machine')]


class DataForm(forms.Form):
    size = forms.ChoiceField(
                                  label='Show full dataframe or head?', choices=HEAD_OR_NOT,
                                  widget=forms.RadioSelect )
    missing_q = forms.ChoiceField(
                                  label='Do you want to check for missing values? If yes, please provide below', choices=YES_NO,
                                  widget=forms.RadioSelect)
    missing_val = forms.DecimalField(localize=False, required=False,
                                     label='How do you want to replace missing values?')
    algo = forms.ChoiceField(
                             label='Which Classifier do you want to use?', choices=ALGOS,
                             widget=forms.RadioSelect)


