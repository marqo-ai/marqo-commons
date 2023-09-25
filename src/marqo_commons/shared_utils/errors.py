from http import HTTPStatus


class InvalidArgError(Exception):
    code = "invalid_argument"
    status_code = HTTPStatus.BAD_REQUEST
    error_type = "invalid_request"
