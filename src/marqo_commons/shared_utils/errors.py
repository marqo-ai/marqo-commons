from http import HTTPStatus


class InvalidSettingsArgError(Exception):
    code = "invalid_argument"
    status_code = HTTPStatus.BAD_REQUEST
    error_type = "invalid_request"
