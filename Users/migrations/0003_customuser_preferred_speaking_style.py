import django.db.models.deletion
from django.db import migrations, models


class Migration(migrations.Migration):
    dependencies = [
        ('vision', '0001_initial'),
        ('Users', '0002_remove_loginhistory_user_delete_emailverification_and_more'),
    ]

    operations = [
        migrations.AddField(
            model_name='customuser',
            name='preferred_speaking_style',
            field=models.ForeignKey(
                blank=True,
                help_text='사용자가 선호하는 응답 화법',
                null=True,
                on_delete=django.db.models.deletion.SET_NULL,
                related_name='preferred_users',
                to='vision.responsestyle',
            ),
        ),
    ]

