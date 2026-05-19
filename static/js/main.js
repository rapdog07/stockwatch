/**
 * StockWatch - Main JavaScript
 *
 * Global UI interactions and utilities.
 */

(function () {
    'use strict';

    // Nav search: redirect to stock page on submit
    document.querySelectorAll('.nav-search').forEach(function (form) {
        form.addEventListener('submit', function (e) {
            const input = form.querySelector('input');
            const ticker = input.value.trim().toUpperCase();
            if (!ticker) {
                e.preventDefault();
                return;
            }
            form.action = '/stock/' + ticker;
        });
    });

    // Any search on homepage
    const heroSearch = document.querySelector('.hero-search');
    if (heroSearch) {
        heroSearch.addEventListener('submit', function (e) {
            const input = heroSearch.querySelector('input');
            const ticker = input.value.trim().toUpperCase();
            if (!ticker) {
                e.preventDefault();
                return;
            }
            heroSearch.action = '/stock/' + ticker;
        });
    }
})();
