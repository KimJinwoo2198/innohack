from django.contrib import admin
from django.utils.translation import gettext_lazy as _
from .models import CustomUser

class CustomUserAdmin(admin.ModelAdmin):
    model = CustomUser

    list_display = (
        'username', 'email', 'phone_number', 'is_active', 
        'last_login'
    )
    list_filter = ('is_active', 'is_staff', 'date_joined')
    search_fields = ('username', 'email', 'phone_number')
    ordering = ('-date_joined',)
    readonly_fields = ('id', 'last_login', 'date_joined')

    fieldsets = (
        (None, {'fields': ('username', 'password')}),
        (_('개인 정보'), {
            'fields': (
                'id', 'email', 'phone_number', 'first_name', 'last_name',
            )
        }),
        (_('계정 보안'), {'fields': ()}),
        (_('중요 날짜'), {'fields': ('last_login', 'date_joined')}),
    )

    add_fieldsets = (
        (None, {
            'classes': ('wide',),
            'fields': (
                'username', 'email', 'phone_number', 'password1', 'password2',
            ),
        }),
    )

admin.site.register(CustomUser, CustomUserAdmin)