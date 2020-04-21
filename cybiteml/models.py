from django.db import models
from django.utils import timezone
from django.contrib.auth.models import User
from django.urls import reverse


def user_directory_path(instance, filename):
    # file will be uploaded to MEDIA_ROOT/user_<id>/<filename>
    return 'files/user_{0}/{1}'.format(instance.creator.id, filename)


class Project(models.Model):
    title = models.CharField(max_length=150)
    description = models.TextField(max_length=500, blank=True)
    file = models.FileField(upload_to=user_directory_path, blank=True)
    date_created = models.DateTimeField(default=timezone.now)
    creator = models.ForeignKey(User, on_delete=models.CASCADE)

    def __str__(self):
        return self.title

    def get_absolute_url(self):
        return reverse('user_projects', kwargs={'username': self.creator})


