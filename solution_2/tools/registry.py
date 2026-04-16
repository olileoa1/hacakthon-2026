from tools.sales_tools import check_tv_package, CHECK_TV_SCHEMA, capture_customer_name, CAPTURE_NAME_SCHEMA

TOOLS_REGISTRY = {
    "check_tv_package": check_tv_package,
    "capture_customer_name": capture_customer_name
}

TOOLS_SCHEMA = [
    CHECK_TV_SCHEMA,
    CAPTURE_NAME_SCHEMA
]
